// The default (and today's only) compute backend: pure-TypeScript matmul, synchronous. Matches
// the math in Ops/MatMul.ts so results are bit-identical.
//
// It deliberately does NOT implement MatMulNtAcc/MatMulTnAcc: Ops/MatMul uses the native backward
// only when BOTH exist, and the transpose fallback is what keeps routed gradients bit-identical
// to the inline path. MatMulNtInto is a forward-only entry, so implementing it changes no gradient.

import type { ComputeBackend } from "./ComputeBackend.ts";
import type { NumArray } from "../Tensor/Tensor.ts";

export class TsBackend implements ComputeBackend {
  MatMul(A: Float64Array, B: Float64Array, M: number, K: number, N: number): Float64Array {
    const Out = new Float64Array(M * N);
    for (let I = 0; I < M; I++) {
      const IK = I * K;
      const IN = I * N;
      for (let P = 0; P < K; P++) {
        const AV = A[IK + P];
        if (AV === 0) continue;
        const PN = P * N;
        for (let J = 0; J < N; J++) Out[IN + J] += AV * B[PN + J];
      }
    }
    return Out;
  }

  /** Out[M,N] = A[M,K] @ Btᵀ with Bt stored [N,K] — row-by-row dots, both operands stride 1. */
  MatMulNtInto(A: NumArray, Bt: NumArray, Out: NumArray, M: number, K: number, N: number): void {
    for (let I = 0; I < M; I++) {
      const IK = I * K;
      const IN = I * N;
      for (let J = 0; J < N; J++) {
        const JK = J * K;
        let Sum = 0;
        for (let P = 0; P < K; P++) Sum += A[IK + P] * Bt[JK + P];
        Out[IN + J] = Sum;
      }
    }
  }
}
