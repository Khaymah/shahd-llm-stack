// Matrix multiply against a TRANSPOSED right operand: A[M,K] @ Bᵀ, with B stored [N,K] -> [M,N].
//   Out[i,j] = sum_p A[i,p] * B[j,p]
// Backward:  dA[M,K] += dOut @ B        (dA[i,p] += sum_j dOut[i,j] * B[j,p])
//            dB[N,K] += dOutᵀ @ A       (dB[j,p] += sum_i dOut[i,j] * A[i,p])
//
// Exists so the two "times a transpose" products never materialize the transpose: the tied LM head
// (X @ Wteᵀ — a Transpose node allocated V×C data + V×C grad every forward, which the Go kernel then
// transposed back) and attention scores (Q @ Kᵀ). Both operands are read along rows (stride 1).
//
// Numerics vs the old Transpose + MatMul pair (F64): every output is the same p-ascending sum from
// zero, every dA cell sums j-ascending, every dB cell sums i-ascending — bit-identical. Under F32
// storage the forward dot product now rounds ONCE (the old ikj loop rounded on every store).
// Compute routing mirrors Ops/MatMul: inline loops when no backend is active, else the backend.

import { Tensor } from "../Tensor/Tensor.ts";
import type { NumArrayCtor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";
import { GetActiveBackend } from "../ComputeBackend/BackendSelector.ts";
import type { ComputeBackend } from "../ComputeBackend/ComputeBackend.ts";
import { AccumulateInto, TransposeBuffer } from "./Internal/MatMulPlumbing.ts";

// Rows of A processed against one row of B while it is hot in L1: the B stream (V×C for the LM
// head) is read once per block instead of once per row. Pure scheduling — per-cell summation order
// is unchanged, so the result does not depend on this value.
const RowBlock = 8;

function InlineForward(A: Tensor, B: Tensor, Out: Tensor, M: number, K: number, N: number): void {
  const AData = A.Data;
  const BData = B.Data;
  const OutData = Out.Data;
  for (let I0 = 0; I0 < M; I0 += RowBlock) {
    const IEnd = Math.min(I0 + RowBlock, M);
    for (let J = 0; J < N; J++) {
      const RowB = J * K;
      for (let I = I0; I < IEnd; I++) {
        const RowA = I * K;
        let Sum = 0;
        for (let P = 0; P < K; P++) Sum += AData[RowA + P] * BData[RowB + P];
        OutData[I * N + J] = Sum;
      }
    }
  }
}

function InlineBackward(A: Tensor, B: Tensor, Out: Tensor, M: number, K: number, N: number): void {
  const AData = A.Data;
  const BData = B.Data;
  const OutGrad = Out.Grad;
  const AGrad = A.Grad;
  // MatMulNT(X, X) (a Gram matrix) aliases the grad buffers; each half must then see the other's
  // contribution only after it finishes, so dB goes through a scratch in that case.
  const Aliased = A === B;
  const BGrad = Aliased ? new (AGrad.constructor as NumArrayCtor)(N * K) : B.Grad;
  for (let I0 = 0; I0 < M; I0 += RowBlock) {
    const IEnd = Math.min(I0 + RowBlock, M);
    for (let J = 0; J < N; J++) {
      const RowB = J * K;
      for (let I = I0; I < IEnd; I++) {
        const G = OutGrad[I * N + J];
        if (G === 0) continue; // e.g. the masked upper triangle of attention scores
        const RowA = I * K;
        for (let P = 0; P < K; P++) {
          AGrad[RowA + P] += G * BData[RowB + P];
          BGrad[RowB + P] += G * AData[RowA + P];
        }
      }
    }
  }
  if (Aliased) AccumulateInto(AGrad, BGrad, N * K);
}

function BackendForward(Backend: ComputeBackend, A: Tensor, B: Tensor, Out: Tensor, M: number, K: number, N: number): void {
  if (Backend.MatMulNtInto !== undefined) {
    Backend.MatMulNtInto(A.Data, B.Data, Out.Data, M, K, N);
  } else if (Backend.MatMulNtAcc !== undefined) {
    Backend.MatMulNtAcc(A.Data, B.Data, Out.Data, M, K, N); // Out is freshly zeroed: += is =
  } else {
    Out.Data.set(Backend.MatMul(A.Data, TransposeBuffer(B.Data, N, K), M, K, N));
  }
}

function BackendBackward(Backend: ComputeBackend, A: Tensor, B: Tensor, Out: Tensor, M: number, K: number, N: number): void {
  const OutGrad = Out.Grad;
  // dA[M,K] += dOut[M,N] @ B[N,K] — a plain NN product (B is already in the layout NN wants).
  AccumulateInto(A.Grad, Backend.MatMul(OutGrad, B.Data, M, N, K), M * K);
  // dB[N,K] += dOutᵀ @ A — the TN kernel consumes dOut transposed in place.
  if (Backend.MatMulTnAcc !== undefined && A !== B) {
    Backend.MatMulTnAcc(OutGrad, A.Data, B.Grad, M, N, K);
  } else {
    AccumulateInto(B.Grad, Backend.MatMul(TransposeBuffer(OutGrad, M, N), A.Data, N, M, K), N * K);
  }
}

/** A[M,K] @ Bᵀ where B is stored [N,K] -> [M,N]. */
export function MatMulNT(A: Tensor, B: Tensor): Tensor {
  const M = A.Rows;
  const K = A.Cols;
  const N = B.Rows;
  if (B.Cols !== K) {
    throw new Error(`MatMulNT: inner dims mismatch ${A.Rows}x${A.Cols} @ (${B.Rows}x${B.Cols})ᵀ`);
  }
  const Out = new Tensor(M, N, undefined, [A, B]);
  const Backend = GetActiveBackend();
  if (Backend === null) InlineForward(A, B, Out, M, K, N);
  else BackendForward(Backend, A, B, Out, M, K, N);

  if (Tape.On) {
    // Same backend for forward and backward (see Ops/MatMul for why a mid-graph switch must not split them).
    const BackwardBackend = Backend;
    Out.BackwardFn = () => {
      if (BackwardBackend === null) InlineBackward(A, B, Out, M, K, N);
      else BackendBackward(BackwardBackend, A, B, Out, M, K, N);
    };
  }
  return Out;
}
