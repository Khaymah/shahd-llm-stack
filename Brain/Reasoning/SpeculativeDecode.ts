// Speculative decoding (Phase 7): a cheap DRAFT model proposes Gamma tokens, and the expensive
// TARGET model verifies all of them in ONE forward pass, accepting the longest correct prefix. The
// win is fewer target forward passes per accepted token when the draft agrees.
//
// GUARANTEE (greedy): the output is BIT-IDENTICAL to plain greedy decoding with the target model,
// as long as the running sequence fits the context window (no BlockSize truncation) — acceptance is
// deterministic in the greedy regime, so this is exactness, not just distribution-matching. Once
// the sequence exceeds BlockSize, both plain and speculative decode over a sliding window and can
// diverge slightly (the verification pass's per-position left context is shorter). Tests stay
// within the window, where the equality is guaranteed.

import type { Shahd } from "../Nn/Shahd.ts";
import type { NumArray } from "../Tensor/Tensor.ts";
import { WithTapeOff } from "../Tensor/Tape.ts";
import { SampleFromLogits } from "../Sampling/Sampler.ts";
import { SeededRng } from "../Random/SeededRng.ts";

export type SpeculativeResult = {
  Ids: number[]; // prompt + generated (== plain greedy within the context window)
  TargetCalls: number; // target forward passes — the expensive metric
  DraftTokens: number; // tokens the draft proposed
  AcceptedTokens: number; // draft proposals the target accepted
};

const Greedy = { Temperature: 0, TopK: 0, TopP: 1 };

// Argmax of a logits row (reuses the greedy branch of SampleFromLogits — the RNG is never consumed).
function ArgmaxAt(Logits: NumArray, Row: number, Vocab: number, Rng: SeededRng): number {
  return SampleFromLogits(Logits, Row * Vocab, Vocab, Greedy, Rng);
}

export function SpeculativeDecodeGreedy(
  Target: Shahd,
  Draft: Shahd,
  PromptIds: number[],
  MaxNewTokens: number,
  Gamma = 4,
): SpeculativeResult {
  return WithTapeOff(() => {
    if (PromptIds.length === 0) throw new Error("SpeculativeDecodeGreedy: empty prompt");
    const Vocab = Target.Config.Model.VocabSize;
    const BlockSize = Target.Config.Model.BlockSize;
    const DraftBlockSize = Draft.Config.Model.BlockSize; // the draft may have a smaller context
    const DummyRng = new SeededRng(0); // greedy => unused
    const Ids = [...PromptIds];
    let TargetCalls = 0;
    let DraftTokens = 0;
    let AcceptedTokens = 0;
    const NewCount = (): number => Ids.length - PromptIds.length;

    while (NewCount() < MaxNewTokens) {
      const G = Math.min(Gamma, BlockSize - 1, MaxNewTokens - NewCount());
      if (G <= 0) break;

      // 1) Draft proposes G tokens greedily.
      const Proposed: number[] = [];
      const Work = [...Ids];
      for (let I = 0; I < G; I++) {
        // Only the draft's last row is read: the head runs for that row alone (ForwardLastLogits).
        const Logits = Draft.ForwardLastLogits(Work.slice(-DraftBlockSize));
        const Tok = ArgmaxAt(Logits.Data, 0, Vocab, DummyRng);
        Proposed.push(Tok);
        Work.push(Tok);
      }
      DraftTokens += G;

      // 2) Target verifies all G in one pass over the extended sequence.
      const Context = [...Ids, ...Proposed].slice(-BlockSize);
      const CtxLen = Context.length;
      const Logits = Target.Forward(Context);
      TargetCalls++;
      const Base = CtxLen - G - 1; // predRow for Proposed[0] (row r predicts Context[r+1])

      let Corrected = false;
      let Accepted = 0;
      for (let I = 0; I < G; I++) {
        const TargetTok = ArgmaxAt(Logits.Data, Base + I, Vocab, DummyRng);
        if (TargetTok === Proposed[I]) {
          Ids.push(Proposed[I]);
          Accepted++;
          AcceptedTokens++;
          if (NewCount() >= MaxNewTokens) break;
        } else {
          Ids.push(TargetTok); // correction at the first divergence
          Corrected = true;
          break;
        }
      }

      // 3) All G accepted with budget left => take the free bonus token (target's own next argmax).
      if (!Corrected && Accepted === G && NewCount() < MaxNewTokens) {
        Ids.push(ArgmaxAt(Logits.Data, CtxLen - 1, Vocab, DummyRng));
      }
    }

    return { Ids: Ids.slice(0, PromptIds.length + MaxNewTokens), TargetCalls, DraftTokens, AcceptedTokens };
  });
}
