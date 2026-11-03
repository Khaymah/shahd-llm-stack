// Argument validation for every native (FFI) matmul entry point. A native kernel trusts its pointers
// and shapes completely: a short buffer is an out-of-bounds read/write in C, and a Float32Array
// handed to an f64 kernel (or vice versa) reads/writes twice/half the bytes it should. Neither is
// catchable once it happens — the best case is garbage numbers, the worst a SIGSEGV or silent heap
// corruption. So every GoFfiBackend method calls ValidateMatMulArgs BEFORE taking a pointer, and the
// checks are cheap (a few comparisons per call, not per element).

import type { NumArray } from "../Tensor/Tensor.ts";

/** Operand layout of a matmul call — decides which product each buffer must hold.
 *  NN: A[M,K] @ B[K,N] -> Out[M,N]           (forward)
 *  NT: A[M,K] @ Btᵀ, Bt stored [N,K] -> Out[M,N]   (dA half / attention scores)
 *  TN: Aᵀ @ DOut, A [M,K], DOut [M,N] -> Db[K,N]   (dB half) */
export type MatMulLayout = "NN" | "NT" | "TN";

// The C kernels take `int` dims (FFIType.i32), so any dim above this would wrap negative natively.
const MaxKernelDim = 2147483647;

function AssertDim(Op: string, Name: string, Value: number): void {
  if (!Number.isInteger(Value) || Value < 1 || Value > MaxKernelDim) {
    throw new Error(`${Op}: dimension ${Name} must be an integer in [1, ${MaxKernelDim}], got ${Value}`);
  }
}

function DescribeBuffer(Buf: unknown): string {
  if (Buf === null || Buf === undefined) return String(Buf);
  const Ctor = (Buf as { constructor?: { name?: string } }).constructor;
  return Ctor?.name ?? typeof Buf;
}

function AssertBuffer(Op: string, Name: string, Buf: NumArray, Required: number, Shape: string): void {
  if (!(Buf instanceof Float64Array) && !(Buf instanceof Float32Array)) {
    throw new Error(`${Op}: ${Name} must be a Float64Array or Float32Array, got ${DescribeBuffer(Buf)}`);
  }
  if (Buf.length === 0) {
    throw new Error(`${Op}: ${Name} is empty (a native kernel cannot take a pointer to zero-length memory)`);
  }
  if (Buf.length < Required) {
    throw new Error(`${Op}: ${Name} has ${Buf.length} elements but shape ${Shape} needs ${Required}`);
  }
}

/** Throws a descriptive Error unless A, B and Out are non-empty typed arrays of ONE element type,
 *  each at least as long as its role in `Layout` requires, and M/K/N are valid kernel dims. */
export function ValidateMatMulArgs(
  Op: string,
  A: NumArray,
  B: NumArray,
  Out: NumArray,
  M: number,
  K: number,
  N: number,
  Layout: MatMulLayout = "NN",
): void {
  AssertDim(Op, "M", M);
  AssertDim(Op, "K", K);
  AssertDim(Op, "N", N);

  const BShape = Layout === "NN" ? { Name: "B", Size: K * N, Shape: `[K=${K},N=${N}]` }
    : Layout === "NT" ? { Name: "Bt", Size: N * K, Shape: `[N=${N},K=${K}]` }
    : { Name: "DOut", Size: M * N, Shape: `[M=${M},N=${N}]` };
  const OutShape = Layout === "TN" ? { Name: "Db", Size: K * N, Shape: `[K=${K},N=${N}]` }
    : { Name: "Out", Size: M * N, Shape: `[M=${M},N=${N}]` };

  AssertBuffer(Op, "A", A, M * K, `[M=${M},K=${K}]`);
  AssertBuffer(Op, BShape.Name, B, BShape.Size, BShape.Shape);
  AssertBuffer(Op, OutShape.Name, Out, OutShape.Size, OutShape.Shape);

  // One element type per call: the kernel is chosen from A, so B/Out must match it exactly.
  if (A.constructor !== B.constructor || A.constructor !== Out.constructor) {
    throw new Error(
      `${Op}: mixed buffer precisions (A=${DescribeBuffer(A)}, ${BShape.Name}=${DescribeBuffer(B)}, ` +
        `${OutShape.Name}=${DescribeBuffer(Out)}) — all operands must share the run's storage precision`,
    );
  }
}
