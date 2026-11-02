// The next-token probability distribution, factored out so both the sampler and speculative sampling
// share ONE implementation of temperature + top-k + top-p (rule #4). ProbsFromLogits returns a
// full-vocab distribution (zeros outside the kept set, renormalized over it); SampleFromDistribution
// draws from it via inverse-CDF in natural (ascending) index order — correct for any fixed
// enumeration order, so no sort is needed. SamplingOptions lives here (the lowest layer that needs
// it); Sampler re-exports it.

import type { SeededRng } from "../Random/SeededRng.ts";
import type { NumArray } from "../Tensor/Tensor.ts";

export type SamplingOptions = {
  Temperature: number; // <= 0 => greedy argmax (honored here: a one-hot distribution at the argmax)
  TopK: number; // 0 => disabled
  TopP: number; // 1 => disabled
};

/** Argmax of a logits row [Offset .. Offset+VocabSize) — the one greedy-selection loop shared by
 *  ProbsFromLogits' Temperature<=0 branch and Sampler.SampleFromLogits' Temperature<=0 branch. */
export function ArgmaxOf(Logits: NumArray, Offset: number, VocabSize: number): number {
  let Best = 0;
  let BestVal = -Infinity;
  for (let J = 0; J < VocabSize; J++) {
    const Val = Logits[Offset + J];
    if (Val > BestVal) {
      BestVal = Val;
      Best = J;
    }
  }
  return Best;
}

/** Temperature-scaled softmax, then top-k, then top-p over the renormalized top-k set (see the ORDER
 *  note below), renormalized over what is finally kept. Temperature<=0
 *  returns a one-hot distribution at the argmax (so every consumer gets the documented greedy). */
export function ProbsFromLogits(Logits: NumArray, Offset: number, VocabSize: number, Options: SamplingOptions): Float64Array {
  if (Options.Temperature <= 0) {
    const Greedy = new Float64Array(VocabSize);
    Greedy[ArgmaxOf(Logits, Offset, VocabSize)] = 1;
    return Greedy;
  }
  const Temperature = Options.Temperature;
  const Probs = new Float64Array(VocabSize);
  let Max = -Infinity;
  for (let J = 0; J < VocabSize; J++) {
    const Scaled = Logits[Offset + J] / Temperature;
    Probs[J] = Scaled;
    if (Scaled > Max) Max = Scaled;
  }
  let Sum = 0;
  for (let J = 0; J < VocabSize; J++) {
    const E = Math.exp(Probs[J] - Max);
    Probs[J] = E;
    Sum += E;
  }
  for (let J = 0; J < VocabSize; J++) Probs[J] /= Sum;

  // Fast path (the common default TopK=0, TopP=1): no truncation, so the full softmax is the answer —
  // skip the O(V log V) sort + renormalization entirely (Probs already sums to 1). Bit-identical.
  if (Options.TopK <= 0 && Options.TopP >= 1) return Probs;

  // ORDER: top-k first, then top-p measured on the top-k set RENORMALIZED (the HF/llama.cpp
  // convention). TopP is a fraction of the mass that survived top-k, not of the full vocabulary:
  // with TopK=2 over probs (0.4, 0.3, ...) and TopP=0.5, the top-1 alone holds 0.4/0.7 ≈ 0.57 of the
  // kept mass, so only it survives. (Measuring against the un-renormalized 0.4 kept both.)
  let Indices: number[] = [];
  for (let J = 0; J < VocabSize; J++) Indices.push(J);
  Indices.sort((A, B) => Probs[B] - Probs[A]);
  if (Options.TopK > 0 && Options.TopK < Indices.length) Indices = Indices.slice(0, Options.TopK);
  if (Options.TopP < 1) {
    let TopKMass = 0;
    for (const Idx of Indices) TopKMass += Probs[Idx];
    // sum(kept)/TopKMass >= TopP, compared as sum(kept) >= TopP*TopKMass (no per-step division).
    const Threshold = Options.TopP * TopKMass;
    const Kept: number[] = [];
    let Cumulative = 0;
    for (const Idx of Indices) {
      Kept.push(Idx);
      Cumulative += Probs[Idx];
      if (Cumulative >= Threshold) break;
    }
    Indices = Kept;
  }

  const Out = new Float64Array(VocabSize);
  let KeptSum = 0;
  for (const Idx of Indices) KeptSum += Probs[Idx];
  if (KeptSum > 0) for (const Idx of Indices) Out[Idx] = Probs[Idx] / KeptSum;
  return Out;
}

/** Sample an index from a probability vector (inverse-CDF walk in natural index order — correct for
 *  any fixed enumeration order, so no sort is needed; one RNG draw). */
export function SampleFromDistribution(Probs: Float64Array, Rng: SeededRng): number {
  let LastNonZero = -1;
  let R = Rng.NextFloat();
  for (let J = 0; J < Probs.length; J++) {
    const P = Probs[J];
    if (P <= 0) continue;
    LastNonZero = J;
    R -= P;
    if (R <= 0) return J;
  }
  return LastNonZero >= 0 ? LastNonZero : 0;
}
