// Matrix multiply: A[M,K] @ B[K,N] -> [M,N]. Backward: dA += dOut @ Bᵀ ; dB += Aᵀ @ dOut.
//
// Compute routing (M2): when a compute backend is active (Config.Compute selects Go FFI / f32 / GPU),
// both the forward and the two backward matmuls route through it; when none is active (the DEFAULT),
// the inline f64 loops run — zero seam overhead, bit-identical, and gradient-checkable. The autograd
// tape always stays here in TypeScript; only the numeric matmul bodies are delegated.

import { Tensor } from "../Tensor/Tensor.ts";
import type { NumArrayCtor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";
import { GetActiveBackend } from "../ComputeBackend/BackendSelector.ts";
import { AccumulateInto, TransposeBuffer } from "./Internal/MatMulPlumbing.ts";

/** Inline backward: dA[M,K] += dOut @ Bᵀ ; dB[K,N] += Aᵀ @ dOut.
 *
 *  I,P,J order: the inner J loop reads B / writes dB with stride 1 (cache-friendly, matching the
 *  forward's ikj order). dA[I,P] lives in a REGISTER accumulator across the J loop instead of being
 *  re-read and re-stored N times — same additions in the same (J-ascending) order, starting from the
 *  stored value, so the result is bit-identical. Under F32 storage the old per-iteration store
 *  rounded to f32 every step, so the accumulator does too (Math.fround) to stay bit-identical. */
function InlineMatMulBackward(A: Tensor, B: Tensor, Out: Tensor, M: number, K: number, N: number): void {
  const AData = A.Data;
  const BData = B.Data;
  const OutGrad = Out.Grad;
  const AGrad = A.Grad;
  // MatMul(X, X) aliases the two grad buffers; dB writes would then land inside a dA cell that the
  // register accumulator is holding. Route dB through a scratch in that (never-in-the-model) case.
  const Aliased = A === B;
  const BGrad = Aliased ? new (AGrad.constructor as NumArrayCtor)(K * N) : B.Grad;
  const RoundToF32 = AGrad instanceof Float32Array;
  for (let I = 0; I < M; I++) {
    const RowOut = I * N;
    for (let P = 0; P < K; P++) {
      const IdxA = I * K + P;
      const AVal = AData[IdxA];
      const RowB = P * N;
      let Acc = AGrad[IdxA];
      for (let J = 0; J < N; J++) {
        const G = OutGrad[RowOut + J];
        if (G === 0) continue;
        Acc = RoundToF32 ? Math.fround(Acc + G * BData[RowB + J]) : Acc + G * BData[RowB + J];
        BGrad[RowB + J] += G * AVal;
      }
      AGrad[IdxA] = Acc;
    }
  }
  if (Aliased) AccumulateInto(AGrad, BGrad, K * N);
}

export function MatMul(A: Tensor, B: Tensor): Tensor {
  const M = A.Rows;
  const K = A.Cols;
  const N = B.Cols;
  if (B.Rows !== K) {
    throw new Error(`MatMul: inner dims mismatch ${A.Rows}x${A.Cols} @ ${B.Rows}x${B.Cols}`);
  }
  const Out = new Tensor(M, N, undefined, [A, B]);

  const Backend = GetActiveBackend();
  if (Backend === null) {
    // Inline f64 fast path (default): bit-identical, gradcheck-exact, no allocation beyond Out.
    const AData = A.Data;
    const BData = B.Data;
    const OutData = Out.Data;
    for (let I = 0; I < M; I++) {
      const RowOut = I * N;
      for (let P = 0; P < K; P++) {
        const AVal = AData[I * K + P];
        if (AVal === 0) continue;
        const RowB = P * N;
        for (let J = 0; J < N; J++) OutData[RowOut + J] += AVal * BData[RowB + J];
      }
    }
  } else if (Backend.MatMulInto !== undefined) {
    Backend.MatMulInto(A.Data, B.Data, Out.Data, M, K, N); // zero-copy: no allocation, no .set copy
  } else {
    Out.Data.set(Backend.MatMul(A.Data, B.Data, M, K, N));
  }

  if (Tape.On) {
    // Capture the SAME backend the forward used, so a mid-graph SetActiveBackend switch cannot
    // desync a node's forward and backward (they must run through one backend/precision).
    const BackwardBackend = Backend;
    Out.BackwardFn = () => {
      if (BackwardBackend === null) {
        InlineMatMulBackward(A, B, Out, M, K, N);
        return;
      }
      if (BackwardBackend.MatMulNtAcc !== undefined && BackwardBackend.MatMulTnAcc !== undefined) {
        // Native backward plumbing: both halves accumulate straight into the grad buffers with
        // ZERO JS-side transposes. dA[M,K] += dOut[M,N] @ Bᵀ — and the stored B[K,N] row-major IS
        // the [outCols=K, inner=N] pre-transposed operand the NT kernel wants, so nothing moves.
        BackwardBackend.MatMulNtAcc(Out.Grad, B.Data, A.Grad, M, N, K);
        // dB[K,N] += Aᵀ @ dOut — the TN kernel reads A's columns in place.
        BackwardBackend.MatMulTnAcc(A.Data, Out.Grad, B.Grad, M, K, N);
        return;
      }
      // Fallback for backends without the native backward halves (e.g. TsBackend, an older DLL):
      // dA = Out.Grad @ Bᵀ  (M,N)@(N,K) -> (M,K)
      const OutGrad = Out.Grad;
      const DA = BackwardBackend.MatMul(OutGrad, TransposeBuffer(B.Data, K, N), M, N, K);
      AccumulateInto(A.Grad, DA, M * K);
      // dB = Aᵀ @ Out.Grad  (K,M)@(M,N) -> (K,N)
      const DB = BackwardBackend.MatMul(TransposeBuffer(A.Data, M, K), OutGrad, K, M, N);
      AccumulateInto(B.Grad, DB, K * N);
    };
  }
  return Out;
}
