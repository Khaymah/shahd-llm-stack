// Softmax cross-entropy over rows: Logits[T,V], Targets length T -> scalar mean loss.
// softmax is FUSED into the loss so the gradient is the numerically-clean (probs - onehot)/T.
// Kept separate from SoftmaxRows (rule #4) — the fused backward is a different mechanism.
//
// The per-row loss is formed in log space: -log p_target = logZ - x_target, with logZ the row's
// log-sum-exp. The old -log(p + 1e-12) floored every loss at ~27.6 nats and stopped agreeing with
// the (unfloored) gradient once p fell below 1e-12; log-sum-exp keeps the two consistent at any p.
// Probs (kept for backward) is allocated in the run's storage precision, like every tensor buffer.

import { Tensor, TensorArrayCtor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";
import { ComputeRowSoftmax } from "./Internal/RowSoftmax.ts";

export function CrossEntropy(Logits: Tensor, Targets: number[]): Tensor {
  const T = Logits.Rows;
  const V = Logits.Cols;
  if (Targets.length !== T) {
    throw new Error(`CrossEntropy: Targets length ${Targets.length} != rows ${T}`);
  }
  const Out = new Tensor(1, 1, undefined, [Logits]);
  const Probs = new (TensorArrayCtor())(T * V);
  const LogitsData = Logits.Data;

  let Loss = 0;
  for (let I = 0; I < T; I++) {
    const Target = Targets[I];
    if (Target < 0 || Target >= V || !Number.isInteger(Target)) {
      throw new Error(`CrossEntropy: target ${Target} at position ${I} out of range [0,${V}) — likely a tokenizer/vocab mismatch`);
    }
    const LogZ = ComputeRowSoftmax(LogitsData, I * V, Probs, I * V, V);
    Loss += LogZ - LogitsData[I * V + Target];
  }
  Out.Data[0] = Loss / T;

  if (Tape.On) {
    Out.BackwardFn = () => {
      const G = Out.Grad[0] / T;
      const LogitsGrad = Logits.Grad;
      for (let I = 0; I < T; I++) {
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
