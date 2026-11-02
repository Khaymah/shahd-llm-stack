// Add a row-vector bias Bias[1,N] to every row of X[M,N]. Backward: dX passes through;
// dBias is the column-sum of dOut over all rows.

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

export function AddBias(X: Tensor, Bias: Tensor): Tensor {
  const M = X.Rows;
  const N = X.Cols;
  if (Bias.Rows !== 1 || Bias.Cols !== N) {
    throw new Error(`AddBias: bias must be [1,${N}], got ${Bias.Rows}x${Bias.Cols}`);
  }
  const Out = new Tensor(M, N, undefined, [X, Bias]);
  const XData = X.Data;
  const BiasData = Bias.Data;
  const OutData = Out.Data;
  for (let I = 0; I < M; I++) {
    for (let J = 0; J < N; J++) OutData[I * N + J] = XData[I * N + J] + BiasData[J];
  }

  if (Tape.On) {
    Out.BackwardFn = () => {
      const OutGrad = Out.Grad;
      const XGrad = X.Grad;
      const BiasGrad = Bias.Grad;
      for (let I = 0; I < M; I++) {
        for (let J = 0; J < N; J++) {
          const G = OutGrad[I * N + J];
          XGrad[I * N + J] += G;
          BiasGrad[J] += G;
        }
      }
    };
  }
  return Out;
}
