// Stochastic speculative sampling (Phase 7, Leviathan et al. / Chen et al.). Unlike the greedy
// variant, this matches the TARGET's sampling distribution exactly (in expectation): the draft
// proposes Gamma tokens from its distribution q; the target computes its distribution p in one pass;
// each draft token is accepted with probability min(1, p/q), and on rejection a replacement is drawn
// from the normalized residual max(0, p - q). If all Gamma are accepted, a bonus token is drawn from
// the target's own distribution. Shares the temperature/top-k/top-p logic with the sampler.
//
// GUARANTEE (verifiable): when the draft IS the target (q == p within the context window), every
// proposal is accepted (min(1, p/q) == 1), so the run is maximally efficient and samples from p.

import type { Shahd } from "../Nn/Shahd.ts";
import { WithTapeOff } from "../Tensor/Tape.ts";
import type { SamplingOptions } from "../Sampling/Sampler.ts";
import { ProbsFromLogits, SampleFromDistribution } from "../Sampling/Distribution.ts";
import type { SeededRng } from "../Random/SeededRng.ts";

export type SpeculativeSampleResult = {
  Ids: number[];
  TargetCalls: number;
  DraftTokens: number;
  AcceptedTokens: number;
};

function ResidualSample(P: Float64Array, Q: Float64Array, Rng: SeededRng): number {
  const Residual = new Float64Array(P.length);
  let Sum = 0;
  for (let J = 0; J < P.length; J++) {
    const R = P[J] - Q[J];
    if (R > 0) {
      Residual[J] = R;
      Sum += R;
    }
  }
  if (Sum <= 0) return SampleFromDistribution(P, Rng); // degenerate: fall back to the target
  for (let J = 0; J < P.length; J++) Residual[J] /= Sum;
  return SampleFromDistribution(Residual, Rng);
}

export function SpeculativeSample(
  Target: Shahd,
  Draft: Shahd,
  PromptIds: number[],
  MaxNewTokens: number,
  Options: SamplingOptions,
  Rng: SeededRng,
  Gamma = 4,
): SpeculativeSampleResult {
  return WithTapeOff(() => {
    if (PromptIds.length === 0) throw new Error("SpeculativeSample: empty prompt");
    const Vocab = Target.Config.Model.VocabSize;
    const BlockSize = Target.Config.Model.BlockSize;
    const DraftBlockSize = Draft.Config.Model.BlockSize; // the draft may have a smaller context
    const Ids = [...PromptIds];
    let TargetCalls = 0;
    let DraftTokens = 0;
    let AcceptedTokens = 0;
    const NewCount = (): number => Ids.length - PromptIds.length;

    while (NewCount() < MaxNewTokens) {
      const G = Math.min(Gamma, BlockSize - 1, MaxNewTokens - NewCount());
      if (G <= 0) break;

      // 1) Draft proposes G tokens from q, keeping each q distribution.
      const Proposed: number[] = [];
      const QDistributions: Float64Array[] = [];
      const Work = [...Ids];
      for (let I = 0; I < G; I++) {
        // Only the draft's last row is read: the head runs for that row alone (ForwardLastLogits).
        const Logits = Draft.ForwardLastLogits(Work.slice(-DraftBlockSize));
        const Q = ProbsFromLogits(Logits.Data, 0, Vocab, Options);
        const Token = SampleFromDistribution(Q, Rng);
        Proposed.push(Token);
        QDistributions.push(Q);
        Work.push(Token);
      }
      DraftTokens += G;

      // 2) Target scores all G positions in one pass.
      const Context = [...Ids, ...Proposed].slice(-BlockSize);
      const CtxLen = Context.length;
      const Logits = Target.Forward(Context);
      TargetCalls++;
      const Base = CtxLen - G - 1; // predRow for Proposed[0]

      let Corrected = false;
      for (let I = 0; I < G; I++) {
        const P = ProbsFromLogits(Logits.Data, (Base + I) * Vocab, Vocab, Options);
        const Q = QDistributions[I];
        const Token = Proposed[I];
        const Ratio = Q[Token] > 0 ? P[Token] / Q[Token] : (P[Token] > 0 ? 1 : 0);
        if (Rng.NextFloat() < Math.min(1, Ratio)) {
          Ids.push(Token);
          AcceptedTokens++;
          if (NewCount() >= MaxNewTokens) break;
        } else {
          Ids.push(ResidualSample(P, Q, Rng));
          Corrected = true;
          break;
        }
      }

      // 3) All accepted with budget left => bonus token from the target's own distribution.
      if (!Corrected && NewCount() < MaxNewTokens) {
        const P = ProbsFromLogits(Logits.Data, (CtxLen - 1) * Vocab, Vocab, Options);
        Ids.push(SampleFromDistribution(P, Rng));
      }
    }

    return { Ids: Ids.slice(0, PromptIds.length + MaxNewTokens), TargetCalls, DraftTokens, AcceptedTokens };
  });
}
