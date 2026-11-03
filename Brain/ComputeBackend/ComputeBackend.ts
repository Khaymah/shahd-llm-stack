// The compute-backend seam (ADR-0002). A backend does pure numeric work on flat Float64 buffers
// with explicit shapes and ZERO knowledge of Tensor/autograd — the only thing that can cheaply
// cross an FFI or process boundary. The autograd tape always stays in TypeScript; only hot-op
// numeric bodies may be routed to a backend.
//
// This SYNC interface fits the model's forward path. TsBackend implements it today. A Go FFI
// backend (in-process, sync) is the target once a working C toolchain exists; the Go SUBPROCESS
// backend (GoBackend) is async and is exposed separately (see the ComputeSpike finding).
//
// The OPTIONAL methods are plumbing for the backward pass: without them, Ops/MatMul must transpose
// B and A on the JS side (single-threaded) and add the returned buffers into the grads with scalar
// `+=` loops — measured overhead worth removing. A backend only declares them when it can honor the
// exact semantics with no hidden copies; Ops/MatMul falls back to the transpose path otherwise, so
// simple backends (TsBackend, TsBackendF32) stay three-line implementations.

import type { NumArray } from "../Tensor/Tensor.ts";

// Buffers arrive in the RUN's storage precision (NumArray = Float64Array | Float32Array). A backend
// must either handle both widths (GoFfiBackend dispatches to the f64/f32 kernel by element type) or
// be routed only under the precision it supports (BackendSelector's job). All buffers of one call
// must share one element type — native backends ENFORCE this (and buffer lengths) per call via
// ValidateMatMulArgs, because a mixed-width or short buffer is an out-of-bounds access in C.
export interface ComputeBackend {
  /** Out[M,N] = A[M,K] @ B[K,N], row-major. Allocates and returns the result. */
  MatMul(A: NumArray, B: NumArray, M: number, K: number, N: number): NumArray;

  /** OPTIONAL zero-copy forward: writes A[M,K] @ B[K,N] into Out (fully overwritten), no allocation. */
  MatMulInto?(A: NumArray, B: NumArray, Out: NumArray, M: number, K: number, N: number): void;

  /** OPTIONAL backward half dA: Out[M,N] += A[M,K] @ Btᵀ, where Bt is stored [N,K] row-major —
   *  i.e. the second operand arrives ALREADY transposed, so the backend must not transpose again. */
  MatMulNtAcc?(A: NumArray, Bt: NumArray, Out: NumArray, M: number, K: number, N: number): void;

  /** OPTIONAL NT forward: Out[M,N] = A[M,K] @ Btᵀ (fully OVERWRITTEN), Bt stored [N,K] row-major —
   *  e.g. attention scores Q @ Kᵀ or a tied LM head (X @ Embᵀ) with no transpose materialized.
   *  Same layout as MatMulNtAcc; only the write mode differs (no zero-fill needed beforehand). */
  MatMulNtInto?(A: NumArray, Bt: NumArray, Out: NumArray, M: number, K: number, N: number): void;

  /** OPTIONAL backward half dB: Db[K,N] += Aᵀ @ DOut, where A is [M,K] and DOut is [M,N] — the
   *  first operand is consumed transposed IN PLACE (no [K,M] copy of A may be materialized). */
  MatMulTnAcc?(A: NumArray, DOut: NumArray, Db: NumArray, M: number, K: number, N: number): void;
}
