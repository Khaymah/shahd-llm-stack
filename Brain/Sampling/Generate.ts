// Autoregressive generation loop (tape off — no autograd graph). Recomputes the forward pass
// over the trailing BlockSize context each step — the uncached reference path (CachedForward.ts is
// the KV-cached twin that matches it token for token). Each step computes the final norm + LM head
// for the LAST position only (Shahd.ForwardLastLogits): the other T-1 rows of the [T, VocabSize]
// logits matrix were never read. Two entry points share one per-token step (DRY):
//   Generate       — synchronous, returns all ids (training samples, tests, non-streaming serving).
//   GenerateAsync  — yields to the event loop between tokens so buffered SSE/WebSocket sends flush,
//                    enabling REAL token-by-token streaming; fires OnToken as each token is produced.

import type { Shahd } from "../Nn/Shahd.ts";
import type { SeededRng } from "../Random/SeededRng.ts";
import type { SamplingOptions } from "./Sampler.ts";
import { SampleFromLogits } from "./Sampler.ts";

// One decode step: forward the trailing context and sample the next token id. ForwardLastLogits
// runs with the tape off (inference).
function StepToken(Model: Shahd, Ids: number[], BlockSize: number, VocabSize: number, Options: SamplingOptions, Rng: SeededRng): number {
  const Logits = Model.ForwardLastLogits(Ids.slice(-BlockSize));
  return SampleFromLogits(Logits.Data, 0, VocabSize, Options, Rng);
}

// An empty prompt has no last position to predict from (the old path read logits at a negative
// offset and sampled garbage) — fail loudly at the entry point instead.
function RequirePrompt(PromptIds: number[], Where: string): void {
  if (PromptIds.length === 0) throw new Error(`${Where}: empty prompt — nothing to condition on`);
}

export function Generate(
  Model: Shahd,
  PromptIds: number[],
  MaxNewTokens: number,
  Options: SamplingOptions,
  Rng: SeededRng,
): number[] {
  RequirePrompt(PromptIds, "Generate");
  const Ids = [...PromptIds];
  const BlockSize = Model.Config.Model.BlockSize;
  const VocabSize = Model.Config.Model.VocabSize;
  for (let S = 0; S < MaxNewTokens; S++) {
    Ids.push(StepToken(Model, Ids, BlockSize, VocabSize, Options, Rng));
  }
  return Ids;
}

export async function GenerateAsync(
  Model: Shahd,
  PromptIds: number[],
  MaxNewTokens: number,
  Options: SamplingOptions,
  Rng: SeededRng,
  OnToken: (Id: number) => void, // fired after each newly sampled token (may throw to abort)
  ShouldStop?: () => boolean, // checked each step; stop early when true (e.g. client disconnected)
): Promise<number[]> {
  RequirePrompt(PromptIds, "GenerateAsync");
  const Ids = [...PromptIds];
  const BlockSize = Model.Config.Model.BlockSize;
  const VocabSize = Model.Config.Model.VocabSize;
  for (let S = 0; S < MaxNewTokens; S++) {
    if (ShouldStop?.() === true) break;
    const Next = StepToken(Model, Ids, BlockSize, VocabSize, Options, Rng);
    Ids.push(Next);
    OnToken(Next); // a throw here (e.g. output safety block) unwinds the loop
    // Yield a macrotask so buffered stream writes reach the client between tokens (real streaming).
    await new Promise<void>((Resolve) => setTimeout(Resolve, 0));
  }
  return Ids;
}
