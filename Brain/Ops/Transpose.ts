// Transpose X[M,N] -> [N,M]. Backward: dX[i,j] += dOut[j,i].

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

export function Transpose(X: Tensor): Tensor {
  const M = X.Rows;
  const N = X.Cols;
  const Out = new Tensor(N, M, undefined, [X]);
  const XData = X.Data;
  const OutData = Out.Data;
  for (let I = 0; I < M; I++) {
    for (let J = 0; J < N; J++) OutData[J * M + I] = XData[I * N + J];
  }

  if (Tape.On) {
    Out.BackwardFn = () => {
      const XGrad = X.Grad;
      const OutGrad = Out.Grad;
      for (let I = 0; I < M; I++) {
        for (let J = 0; J < N; J++) XGrad[I * N + J] += OutGrad[J * M + I];
      }
    };
  }
  return Out;
}
