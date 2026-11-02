// Row-wise softmax on X[M,N] (max-subtracted for numerical stability). Backward is the exact
// softmax Jacobian-vector product: dX_j = p_j * (dOut_j - sum_k p_k*dOut_k). Kept SEPARATE
// from CrossEntropy (rule #4): although both involve softmax, their backward passes are
// genuinely different mechanisms. (Attention now uses the fused CausalSoftmax; this op stays
// the general row softmax and the unfused reference for it.)

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";
import { ComputeRowSoftmax } from "./Internal/RowSoftmax.ts";

export function SoftmaxRows(X: Tensor): Tensor {
  const M = X.Rows;
  const N = X.Cols;
  const Out = new Tensor(M, N, undefined, [X]);
  const XData = X.Data;
  const OutData = Out.Data;

  for (let I = 0; I < M; I++) {
    ComputeRowSoftmax(XData, I * N, OutData, I * N, N);
  }

  if (Tape.On) {
    Out.BackwardFn = () => {
      const P = Out.Data;
      const OutGrad = Out.Grad;
      const XGrad = X.Grad;
      for (let I = 0; I < M; I++) {
        const Row = I * N;
        let Dot = 0;
        for (let J = 0; J < N; J++) Dot += P[Row + J] * OutGrad[Row + J];
        for (let J = 0; J < N; J++) {
          XGrad[Row + J] += P[Row + J] * (OutGrad[Row + J] - Dot);
        }
      }
    };
  }
  return Out;
}
