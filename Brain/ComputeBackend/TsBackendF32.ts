// CPU Float32 compute backend (M2, the mixed-precision path + the GPU prerequisite). It COPIES the
// caller's f64 inputs into fresh Float32Arrays (it never aliases or mutates them — the autograd
// backward reads those f64 originals), accumulates the matmul in f32, and widens the result back to
// Float64Array. This is the numerical reference a WebGPU f32 kernel (M5) should match.
//
// NOTE: results are f32-precision, so this path is NOT gradient-checkable — its correctness rests on
// the parity test against TsBackend, and F64 stays the default for any run whose exactness matters.

import type { ComputeBackend } from "./ComputeBackend.ts";
import type { NumArray } from "../Tensor/Tensor.ts";

export class TsBackendF32 implements ComputeBackend {
  MatMul(A: Float64Array, B: Float64Array, M: number, K: number, N: number): Float64Array {
    const Af = Float32Array.from(A); // copies; A is left untouched
    const Bf = Float32Array.from(B);
    const Out = new Float32Array(M * N); // f32 accumulator (each += rounds to f32)
    for (let I = 0; I < M; I++) {
      const IK = I * K;
      const IN = I * N;
      for (let P = 0; P < K; P++) {
        const AV = Af[IK + P];
        if (AV === 0) continue;
        const PN = P * N;
        for (let J = 0; J < N; J++) Out[IN + J] += AV * Bf[PN + J];
      }
    }
    return Float64Array.from(Out);
  }

  /** Out[M,N] = A[M,K] @ Btᵀ (Bt stored [N,K]), with operands and the running sum rounded to f32
   *  exactly like MatMul above. Writes into the caller's Out (whatever its element type). */
  MatMulNtInto(A: NumArray, Bt: NumArray, Out: NumArray, M: number, K: number, N: number): void {
    const Af = Float32Array.from(A.subarray(0, M * K));
    const Bf = Float32Array.from(Bt.subarray(0, N * K));
    const Sum = new Float32Array(1); // one-slot f32 accumulator: every += rounds to f32
    for (let I = 0; I < M; I++) {
      const IK = I * K;
      const IN = I * N;
      for (let J = 0; J < N; J++) {
        const JK = J * K;
        Sum[0] = 0;
        for (let P = 0; P < K; P++) Sum[0] += Af[IK + P] * Bf[JK + P];
        Out[IN + J] = Sum[0];
      }
    }
  }
}
