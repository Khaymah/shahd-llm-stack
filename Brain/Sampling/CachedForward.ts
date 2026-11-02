// Incremental (KV-cached) forward + generation — the inference-only twin of the training forward
// (deliberately a separate path, not a flag on the training forward). It processes ONE token row
// in plain numerics (tape off), attending over the growing KvCache instead of recomputing past
// K/V. Supports every architecture the config can select: learned or rotary positions, LayerNorm
// or RMSNorm, ReLU / SwiGLU / GeGLU MLPs, and MHA or GQA. Verified against the Tensor forward by
// CachedForward.Test.ts (per-step logits within 1e-9, greedy ids identical to Generate).
//
// CONTEXT OVERFLOW mirrors Generate exactly: Generate always conditions on the trailing BlockSize
// tokens, positions restarting at 0 at the window start. When the cache is full, the window has
// slid, so every cached K/V (each depends on its own left context) is stale; the decoder rebuilds
// the cache from the last BlockSize-1 tokens with ONE batched blocks-only forward
// (Shahd.ForwardHidden + KvSink) and then steps the new token. Past BlockSize each token therefore
// costs one blocks-only window forward — the price of matching Generate token-for-token.

import type { Shahd } from "../Nn/Shahd.ts";
import type { SeededRng } from "../Random/SeededRng.ts";
import type { SamplingOptions } from "./Sampler.ts";
import { KvCache } from "./KvCache.ts";
import { SampleFromLogits } from "./Sampler.ts";
import { WithTapeOff } from "../Tensor/Tape.ts";
import { RopeCosSin, RopeFrequencies, RotateRow } from "../Ops/OpsBarrel.ts";
import { AttendRow, HeadLogitsRow, MatMulRow, MlpRow, NormRow } from "./CachedRowMath.ts";

function ValidateTokenId(TokenId: number, VocabSize: number, Where: string): void {
  if (!Number.isInteger(TokenId) || TokenId < 0 || TokenId >= VocabSize) {
    throw new Error(`${Where}: token id ${TokenId} out of range [0,${VocabSize}) — likely a tokenizer/vocab mismatch`);
  }
}

/** Run one token through every block at window position Position (== Cache.Length), appending its
 *  K/V to the cache. Returns the final hidden row (before the final norm). */
function AdvanceOneToken(Model: Shahd, Cache: KvCache, TokenId: number, Position: number): Float64Array {
  const Cfg = Model.Config;
  ValidateTokenId(TokenId, Cfg.Model.VocabSize, "CachedForwardStep");
  if (Position !== Cache.Length) {
    throw new Error(`CachedForwardStep: position ${Position} must equal the cached length ${Cache.Length} (positions are contiguous)`);
  }
  if (Position >= Cache.Capacity) {
    throw new Error(`CachedForwardStep: position ${Position} is outside the ${Cache.Capacity}-token window — rebuild the cache (see CachedDecoder)`);
  }
  const E = Cfg.Model.EmbedDim;
  const HeadDim = Cfg.Derived.HeadDim;
  const Scale = Cfg.Derived.AttentionScale;
  const Hidden = Cfg.Derived.MlpHidden;
  const Wte = Model.Embedding.Wte.Data;
  const Wpe = Model.Embedding.Wpe;

  const X = new Float64Array(E);
  for (let J = 0; J < E; J++) X[J] = Wte[TokenId * E + J];
  if (Wpe !== null) {
    if (Position >= Wpe.Rows) throw new Error(`CachedForwardStep: position ${Position} has no learned embedding (Wpe has ${Wpe.Rows} rows)`);
    for (let J = 0; J < E; J++) X[J] += Wpe.Data[Position * E + J];
  }

  // RoPE: one (cos, sin) row for this position, shared by every layer's Q and K.
  const UseRope = Cfg.Model.PositionScheme === "Rope";
  const Half = HeadDim / 2;
  const Rope = UseRope ? RopeCosSin(RopeFrequencies(HeadDim), Position, 1) : null;
  const Rotate = (Row: Float64Array): void => {
    if (Rope !== null) RotateRow(Row, 0, Row, 0, Rope.Cos, Rope.Sin, 0, Half);
  };

  for (let L = 0; L < Model.Blocks.length; L++) {
    const Block = Model.Blocks[L];
    const Attn = Block.Attn;
    const Xn = NormRow(Block.Ln1, X);

    // Each KV head is projected once and cached (K after rotation); its query group reads it.
    for (let Kv = 0; Kv < Attn.KvHeads; Kv++) {
      const K = MatMulRow(Xn, Attn.WkHeads[Kv].Data, E, HeadDim);
      Rotate(K);
      Cache.Append(L, Kv, K, MatMulRow(Xn, Attn.WvHeads[Kv].Data, E, HeadDim));
    }

    const AttnOut = new Float64Array(E);
    for (let H = 0; H < Attn.NumHeads; H++) {
      const KvIndex = Math.floor(H / Attn.GroupSize); // same grouping as MultiHeadAttention
      const Q = MatMulRow(Xn, Attn.WqHeads[H].Data, E, HeadDim);
      Rotate(Q);
      const Ctx = AttendRow(Q, Cache.GetKeys(L, KvIndex), Cache.GetValues(L, KvIndex), HeadDim, Scale);
      const Proj = MatMulRow(Ctx, Attn.WoHeads[H].Data, HeadDim, E);
      for (let J = 0; J < E; J++) AttnOut[J] += Proj[J];
    }
    for (let J = 0; J < E; J++) X[J] += AttnOut[J]; // residual

    const MlpOut = MlpRow(Block.Mlp, NormRow(Block.Ln2, X), Hidden);
    for (let J = 0; J < E; J++) X[J] += MlpOut[J]; // residual
  }
  return X;
}

/** Process one token at window position Position (must equal Cache.Length), updating the cache;
 *  returns next-token logits [VocabSize]. */
export function CachedForwardStep(Model: Shahd, Cache: KvCache, TokenId: number, Position: number): Float64Array {
  return HeadLogitsRow(Model, AdvanceOneToken(Model, Cache, TokenId, Position));
}

/** Reset the cache and fill it with Ids (window positions 0..len-1) in ONE batched blocks-only
 *  forward — no LM head, and the tensor matmuls (or the active compute backend) do the work. */
export function PrefillCache(Model: Shahd, Cache: KvCache, Ids: number[]): void {
  Cache.Reset();
  if (Ids.length === 0) return;
  for (const Id of Ids) ValidateTokenId(Id, Model.Config.Model.VocabSize, "PrefillCache");
  if (Ids.length > Cache.Capacity) {
    throw new Error(`PrefillCache: ${Ids.length} tokens exceed the cache capacity ${Cache.Capacity}`);
  }
  WithTapeOff(() => {
    Model.ForwardHidden(Ids, (Layer, KvIndex, K, V) => Cache.AppendRows(Layer, KvIndex, K.Data, V.Data, K.Rows));
  });
}

/** Stateful decoder over the sliding window: holds the cache and the logits for the next token. */
export class CachedDecoder {
  private readonly Model: Shahd;
  private readonly Cache: KvCache;
  private readonly Window: number[]; // the ids currently in the cache, oldest first
  /** Logits [VocabSize] for the token that follows everything pushed so far. */
  Logits: Float64Array;

  constructor(Model: Shahd, PromptIds: number[]) {
    if (PromptIds.length === 0) throw new Error("CachedDecoder: empty prompt — nothing to condition on");
    this.Model = Model;
    this.Cache = KvCache.ForModel(Model);
    const BlockSize = Model.Config.Model.BlockSize;
    const Context = PromptIds.slice(-BlockSize); // same trailing window Generate uses
    const Last = Context[Context.length - 1];
    PrefillCache(Model, this.Cache, Context.slice(0, -1));
    this.Window = Context.slice(0, -1);
    this.Logits = this.Step(Last);
  }

  /** Append the sampled token and compute the logits for the one after it. */
  Push(TokenId: number): void {
    ValidateTokenId(TokenId, this.Model.Config.Model.VocabSize, "CachedDecoder.Push"); // before any rebuild work
    if (this.Cache.IsFull) {
      // The window slides: rebuild from the last BlockSize-1 ids, then step the new one.
      const Keep = this.Window.slice(-(this.Cache.Capacity - 1));
      PrefillCache(this.Model, this.Cache, Keep);
      this.Window.length = 0;
      this.Window.push(...Keep);
    }
    this.Logits = this.Step(TokenId);
  }

  private Step(TokenId: number): Float64Array {
    const Logits = WithTapeOff(() => CachedForwardStep(this.Model, this.Cache, TokenId, this.Cache.Length));
    this.Window.push(TokenId);
    return Logits;
  }
}

/** KV-cached autoregressive generation — same ids as Generate (same window, same sampling draws). */
export function CachedGenerate(
  Model: Shahd,
  PromptIds: number[],
  MaxNewTokens: number,
  Options: SamplingOptions,
  Rng: SeededRng,
): number[] {
  if (PromptIds.length === 0) throw new Error("CachedGenerate: empty prompt — nothing to condition on");
  if (!(MaxNewTokens >= 1)) return [...PromptIds]; // also skips the prefill entirely
  const Decoder = new CachedDecoder(Model, PromptIds);
  const VocabSize = Model.Config.Model.VocabSize;
  const Ids = [...PromptIds];
  for (let S = 0; S < MaxNewTokens; S++) {
    const Next = SampleFromLogits(Decoder.Logits, 0, VocabSize, Options, Rng);
    Ids.push(Next);
    if (S + 1 < MaxNewTokens) Decoder.Push(Next); // the last token's successor is never needed
  }
  return Ids;
}

/** Streaming twin of CachedGenerate with GenerateAsync's contract: OnToken fires per new token (a
 *  throw aborts), ShouldStop is polled each step, and the loop yields a macrotask between tokens so
 *  buffered SSE/WebSocket writes flush. Lets streaming serving use the cache. */
export async function CachedGenerateAsync(
  Model: Shahd,
  PromptIds: number[],
  MaxNewTokens: number,
  Options: SamplingOptions,
  Rng: SeededRng,
  OnToken: (Id: number) => void,
  ShouldStop?: () => boolean,
): Promise<number[]> {
  if (PromptIds.length === 0) throw new Error("CachedGenerateAsync: empty prompt — nothing to condition on");
  if (!(MaxNewTokens >= 1)) return [...PromptIds];
  const Decoder = new CachedDecoder(Model, PromptIds);
  const VocabSize = Model.Config.Model.VocabSize;
  const Ids = [...PromptIds];
  for (let S = 0; S < MaxNewTokens; S++) {
    if (ShouldStop?.() === true) break;
    const Next = SampleFromLogits(Decoder.Logits, 0, VocabSize, Options, Rng);
    Ids.push(Next);
    OnToken(Next);
    if (S + 1 < MaxNewTokens) Decoder.Push(Next);
    await new Promise<void>((Resolve) => setTimeout(Resolve, 0));
  }
  return Ids;
}
