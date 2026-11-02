// Causal mask on a square score matrix X[T,T]: positions j>i (the future) become -1e9 before
// softmax. Backward: masked positions are constants (no dependence on X), so they get zero
// gradient — gradient flows only where j<=i. Getting this right is a classic first-attempt bug.
// (Attention now uses the fused CausalSoftmax; this op stays as the unfused reference.)

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

const NegInf = -1e9;

export function CausalMask(X: Tensor): Tensor {
  const T = X.Rows;
  if (X.Cols !== T) {
    throw new Error(`CausalMask: expected a square [T,T] matrix, got ${X.Rows}x${X.Cols}`);
  }
  const Out = new Tensor(T, T, undefined, [X]);
  const XData = X.Data;
  const OutData = Out.Data;
  for (let I = 0; I < T; I++) {
    for (let J = 0; J < T; J++) OutData[I * T + J] = J > I ? NegInf : XData[I * T + J];
  }

  if (Tape.On) {
    Out.BackwardFn = () => {
      const OutGrad = Out.Grad;
      const XGrad = X.Grad;
      for (let I = 0; I < T; I++) {
        for (let J = 0; J < T; J++) {
          if (J <= I) XGrad[I * T + J] += OutGrad[I * T + J];
        }
      }
    };
  }
  return Out;
}
