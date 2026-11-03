// The full Shahd model: token/position embeddings -> N transformer blocks -> final norm
// -> LM head producing next-token logits [T, VocabSize]. Supports weight tying (LM head =
// transpose of the token embedding, applied via MatMulNT) and asserts the context bound (closes
// the Wpe OOB risk). ForwardLastLogits is the inference entry that skips the head for all but the
// last row; ForwardHidden exposes the block stack (and per-layer K/V) to the KV-cached decoder.

import type { Tensor } from "../Tensor/Tensor.ts";
import type { SeededRng } from "../Random/SeededRng.ts";
import type { ResolvedConfig } from "../Config/ConfigTypes.ts";
import { MatMul, MatMulNT, AddBias } from "../Ops/OpsBarrel.ts";
import { FromArray, Zeros } from "../Tensor/TensorFactories.ts";
import { WithTapeOff } from "../Tensor/Tape.ts";
import { Embedding } from "./Embedding.ts";
import { Block } from "./Block.ts";
import { NormLayer } from "./NormLayer.ts";
import { InitWeight } from "./InitPolicy.ts";

/** Per-layer KV observer for ForwardHidden (see MultiHeadAttention.KvSink). */
export type LayerKvSink = (Layer: number, KvIndex: number, K: Tensor, V: Tensor) => void;

export class Shahd {
  Config: ResolvedConfig;
  Embedding: Embedding;
  Blocks: Block[] = [];
  LnFinal: NormLayer;
  LmHead: Tensor | null; // null when weights are tied (head = transpose of token embedding)
  LmHeadBias: Tensor; // [1, VocabSize]
  WeightTying: boolean;

  constructor(Config: ResolvedConfig, Rng: SeededRng) {
    const { EmbedDim, VocabSize, NumLayers } = Config.Model;
    this.Config = Config;
    this.Embedding = new Embedding(Config, Rng);
    for (let L = 0; L < NumLayers; L++) this.Blocks.push(new Block(Config, Rng));
    this.LnFinal = new NormLayer(EmbedDim, Config);
    this.WeightTying = Config.Model.WeightTying;
    this.LmHead = this.WeightTying ? null : InitWeight(EmbedDim, VocabSize, Rng, Config);
    this.LmHeadBias = Zeros(1, VocabSize);
  }

  /** ids[T] -> hidden states [T, EmbedDim] after the last block (before the final norm). The
   *  optional Sink observes every layer's per-KV-head K (post-RoPE) and V as they are built. */
  ForwardHidden(Ids: number[], Sink?: LayerKvSink): Tensor {
    if (Ids.length > this.Config.Model.BlockSize) {
      throw new Error(
        `Shahd.Forward: sequence length ${Ids.length} exceeds BlockSize ${this.Config.Model.BlockSize}`,
      );
    }
    let X = this.Embedding.Forward(Ids);
    for (let L = 0; L < this.Blocks.length; L++) {
      const LayerSink = Sink === undefined ? undefined : (Kv: number, K: Tensor, V: Tensor): void => Sink(L, Kv, K, V);
      X = this.Blocks[L].Forward(X, LayerSink);
    }
    return X;
  }

  /** ids[T] -> logits[T, VocabSize]. */
  Forward(Ids: number[]): Tensor {
    return this.Head(this.LnFinal.Forward(this.ForwardHidden(Ids)));
  }

  /** Next-token logits [1, VocabSize] for the LAST position only — inference (tape off, no graph).
   *  The blocks still run over the whole context (every position feeds attention), but the final
   *  norm and the LM head — a [T, VocabSize] product, the single largest matmul at V >> C — run for
   *  one row. Per-row math is unchanged, so this equals the last row of Forward(Ids) exactly. */
  ForwardLastLogits(Ids: number[]): Tensor {
    if (Ids.length === 0) throw new Error("Shahd.ForwardLastLogits: empty context — nothing to predict from");
    return WithTapeOff(() => {
      const Hidden = this.ForwardHidden(Ids);
      const E = Hidden.Cols;
      const Last = FromArray(1, E, Hidden.Data.slice((Hidden.Rows - 1) * E, Hidden.Rows * E));
      return this.Head(this.LnFinal.Forward(Last));
    });
  }

  /** Normalized hidden [T, EmbedDim] -> logits [T, VocabSize]. */
  private Head(X: Tensor): Tensor {
    if (this.WeightTying) {
      // Tied head: logits = X @ Wteᵀ, read straight from the [VocabSize, EmbedDim] table (no transpose).
      return AddBias(MatMulNT(X, this.Embedding.Wte), this.LmHeadBias);
    }
    if (this.LmHead === null) throw new Error("Shahd: LmHead missing on an untied model");
    return AddBias(MatMul(X, this.LmHead), this.LmHeadBias);
  }

  /** All trainable parameters in a stable order (consumed by the optimizer and checkpoints). */
  Parameters(): Tensor[] {
    const Params: Tensor[] = [...this.Embedding.Parameters()];
    for (const CurrentBlock of this.Blocks) Params.push(...CurrentBlock.Parameters());
    Params.push(...this.LnFinal.Parameters());
    if (this.LmHead !== null) Params.push(this.LmHead);
    Params.push(this.LmHeadBias);
    return Params;
  }
}
