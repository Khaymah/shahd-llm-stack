// Softmax cross-entropy over rows, but only positions where Mask[i] is true contribute (loss and
// gradient). This is the SFT loss: we train ONLY on the assistant's response tokens, not the
// system/user prompt (Phase 4). Reduces to CrossEntropy when every position is masked-in.
// Loss is formed in log space (logZ - x_target) and Probs is stored in the run's precision —
// see CrossEntropy.ts for why. Masked-out rows are never softmaxed (their Probs stay unused).

import { Tensor, TensorArrayCtor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";
import { ComputeRowSoftmax } from "./Internal/RowSoftmax.ts";

export function MaskedCrossEntropy(Logits: Tensor, Targets: number[], Mask: boolean[]): Tensor {
  const T = Logits.Rows;
  const V = Logits.Cols;
  if (Targets.length !== T || Mask.length !== T) {
    throw new Error(`MaskedCrossEntropy: Targets/Mask length must equal rows ${T}`);
  }
  const Out = new Tensor(1, 1, undefined, [Logits]);
  const Probs = new (TensorArrayCtor())(T * V);
  const LogitsData = Logits.Data;

  let Count = 0;
  let Loss = 0;
  for (let I = 0; I < T; I++) {
    if (!Mask[I]) continue;
    const Target = Targets[I];
    if (Target < 0 || Target >= V || !Number.isInteger(Target)) {
      throw new Error(`MaskedCrossEntropy: target ${Target} at position ${I} out of range [0,${V}) — likely a tokenizer/vocab mismatch`);
    }
    const LogZ = ComputeRowSoftmax(LogitsData, I * V, Probs, I * V, V);
    Loss += LogZ - LogitsData[I * V + Target];
    Count++;
  }
  if (Count === 0) {
    throw new Error("MaskedCrossEntropy: Mask has no true positions - nothing to train on; check the caller's masking logic before calling");
  }
  const Denom = Count;
  Out.Data[0] = Loss / Denom;

  if (Tape.On) {
    Out.BackwardFn = () => {
      const G = Out.Grad[0] / Denom;
      const LogitsGrad = Logits.Grad;
      for (let I = 0; I < T; I++) {
        if (!Mask[I]) continue;
        const Row = I * V;
        const Target = Targets[I];
        for (let J = 0; J < V; J++) {
          LogitsGrad[Row + J] += G * (Probs[Row + J] - (J === Target ? 1 : 0));
        }
      }
    };
  }
  return Out;
}
