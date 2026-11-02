// Fused causal attention softmax over raw scores X[T,T]:
//   P[i,j] = softmax_j( Factor * X[i,j] )  over j <= i,      P[i,j] = 0 for j > i (the future).
// One op replaces Scale -> CausalMask -> SoftmaxRows. The unfused chain kept FOUR T×T tensors per
// head on the tape (scores, scaled, masked, probs) — at T=256 that was ~75 MB of activations per
// sequence, growing as T². This keeps ONE (P), and never touches the masked upper triangle.
//
// Backward (softmax JVP restricted to the visible triangle, chained through the scale):
//   dX[i,j] = Factor * P[i,j] * (dP[i,j] - sum_{k<=i} P[i,k] * dP[i,k])   for j <= i, else 0.
// In F64 this is bit-identical to the unfused chain: the masked entries contributed exact zeros
// there (exp(-1e9 - max) == 0), and every surviving term is the same expression in the same order.
// Tests/FusedAttention.Test.ts holds the two paths against each other, forward and backward.

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";
import { ComputeRowSoftmax } from "./Internal/RowSoftmax.ts";

export function CausalSoftmax(X: Tensor, Factor: number): Tensor {
  const T = X.Rows;
  if (X.Cols !== T) {
    throw new Error(`CausalSoftmax: expected a square [T,T] score matrix, got ${X.Rows}x${X.Cols}`);
  }
  const Out = new Tensor(T, T, undefined, [X]);
  const XData = X.Data;
  const P = Out.Data;

  for (let I = 0; I < T; I++) {
    const Row = I * T;
    const Visible = I + 1; // columns 0..I
    for (let J = 0; J < Visible; J++) P[Row + J] = XData[Row + J] * Factor;
    // In place over the visible prefix (the helper reads each input before writing that slot);
    // the upper triangle stays at the buffer's initial zeros.
    ComputeRowSoftmax(P, Row, P, Row, Visible);
  }

  if (Tape.On) {
    Out.BackwardFn = () => {
      const Probs = Out.Data;
      const OutGrad = Out.Grad;
      const XGrad = X.Grad;
      for (let I = 0; I < T; I++) {
        const Row = I * T;
        const Visible = I + 1;
        let Dot = 0;
        for (let J = 0; J < Visible; J++) Dot += Probs[Row + J] * OutGrad[Row + J];
        for (let J = 0; J < Visible; J++) {
          XGrad[Row + J] += Factor * (Probs[Row + J] * (OutGrad[Row + J] - Dot));
        }
      }
    };
  }
  return Out;
}
