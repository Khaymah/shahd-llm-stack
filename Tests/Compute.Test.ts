import { test, expect, afterEach } from "bun:test";
import { TsBackend } from "../Brain/ComputeBackend/TsBackend.ts";
import { TsBackendF32 } from "../Brain/ComputeBackend/TsBackendF32.ts";
import {
  GetActiveBackend,
  SetActiveBackend,
  ActivateFromConfig,
} from "../Brain/ComputeBackend/BackendSelector.ts";
import { MatMul } from "../Brain/Ops/MatMul.ts";
import { Tensor } from "../Brain/Tensor/Tensor.ts";
import { SeededRng } from "../Brain/Random/SeededRng.ts";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";

// STRUCTURAL RESET (advisor): never let a thrown assertion leave a non-default backend active and
// pollute the gradcheck-bearing tests elsewhere in the suite.
afterEach(() => SetActiveBackend(null));

function RandTensor(Rows: number, Cols: number, Rng: SeededRng): Tensor {
  const T = new Tensor(Rows, Cols);
  for (let I = 0; I < T.Size; I++) T.Data[I] = Rng.NextGaussian();
  return T;
}

test("default active backend is null (the inline f64 fast path)", () => {
  expect(GetActiveBackend()).toBe(null);
});

test("TsBackendF32 matches TsBackend within f32 tolerance and leaves inputs untouched", () => {
  const Rng = new SeededRng(1);
  const A = new Float64Array(64 * 48);
  const B = new Float64Array(48 * 32);
  for (let I = 0; I < A.length; I++) A[I] = Rng.NextGaussian();
  for (let I = 0; I < B.length; I++) B[I] = Rng.NextGaussian();
  const ASnapshot = Float64Array.from(A);
  const Exact = new TsBackend().MatMul(A, B, 64, 48, 32);
  const F32 = new TsBackendF32().MatMul(A, B, 64, 48, 32);
  let MaxRel = 0;
  for (let I = 0; I < Exact.length; I++) {
    MaxRel = Math.max(MaxRel, Math.abs(Exact[I] - F32[I]) / (Math.abs(Exact[I]) + 1e-6));
  }
  expect(MaxRel).toBeLessThan(1e-3);
  expect(A).toEqual(ASnapshot); // f32 path copied inputs, did not mutate them
});

test("routing MatMul through TsBackend is bit-identical to the inline path", () => {
  const Rng = new SeededRng(2);
  const A = RandTensor(10, 7, Rng);
  const B = RandTensor(7, 5, Rng);
  const Inline = MatMul(A, B); // Active === null
  SetActiveBackend(new TsBackend());
  const Routed = MatMul(A, B);
  for (let I = 0; I < Inline.Size; I++) expect(Routed.Data[I]).toBe(Inline.Data[I]);
});

test("routing MatMul through TsBackendF32 stays within f32 tolerance, then toggles back", () => {
  const Rng = new SeededRng(3);
  const A = RandTensor(12, 9, Rng);
  const B = RandTensor(9, 6, Rng);
  const Exact = MatMul(A, B);
  SetActiveBackend(new TsBackendF32());
  const F32 = MatMul(A, B);
  let MaxRel = 0;
  for (let I = 0; I < Exact.Size; I++) MaxRel = Math.max(MaxRel, Math.abs(Exact.Data[I] - F32.Data[I]) / (Math.abs(Exact.Data[I]) + 1e-6));
  expect(MaxRel).toBeLessThan(1e-3);
  SetActiveBackend(null); // runtime toggle back to CPU f64
  const Back = MatMul(A, B);
  for (let I = 0; I < Exact.Size; I++) expect(Back.Data[I]).toBe(Exact.Data[I]);
});

test("routed backward gradients match the inline backward within f64 rounding", () => {
  const Rng = new SeededRng(4);
  const A = RandTensor(6, 5, Rng);
  const B = RandTensor(5, 4, Rng);
  const InlineOut = MatMul(A, B);
  for (let I = 0; I < InlineOut.Size; I++) InlineOut.Grad[I] = 1;
  InlineOut.BackwardFn?.();
  const InlineDA = Float64Array.from(A.Grad);
  const InlineDB = Float64Array.from(B.Grad);

  A.Grad.fill(0);
  B.Grad.fill(0);
  SetActiveBackend(new TsBackend());
  const RoutedOut = MatMul(A, B);
  for (let I = 0; I < RoutedOut.Size; I++) RoutedOut.Grad[I] = 1;
  RoutedOut.BackwardFn?.();
  for (let I = 0; I < A.Size; I++) expect(Math.abs(A.Grad[I] - InlineDA[I])).toBeLessThan(1e-9);
  for (let I = 0; I < B.Size; I++) expect(Math.abs(B.Grad[I] - InlineDB[I])).toBeLessThan(1e-9);
});

test("ActivateFromConfig honors the config and falls back to CPU when a backend is missing", () => {
  // Test backends EXPLICITLY (not via the default) so this stays portable regardless of which backend
  // is the configured default and whether the Go FFI DLL is present on the CI machine.
  const Ts = ActivateFromConfig(LoadConfig({ Overrides: { Compute: { Backend: "Ts" } }, UseCli: false, UseEnv: false }));
  expect(Ts.Chosen).toContain("Ts/F64");
  expect(GetActiveBackend()).toBe(null); // Ts f64 => inline

  const F32 = ActivateFromConfig(LoadConfig({ Overrides: { Compute: { Backend: "Ts", Precision: "F32" } }, UseCli: false, UseEnv: false }));
  expect(F32.Chosen).toContain("F32");
  // F32 is real STORAGE now (SetTensorPrecision), not a per-call conversion backend: the inline
  // loops are precision-agnostic, so Ts/F32 also runs inline. (The Gpu activation below restores
  // the F64 default for the rest of the suite.)
  expect(GetActiveBackend()).toBe(null);

  // GPU is not built; with FallbackToCpu it must not throw and must drop to CPU.
  const Gpu = ActivateFromConfig(LoadConfig({ Overrides: { Compute: { Backend: "Gpu", FallbackToCpu: true } }, UseCli: false, UseEnv: false }));
  expect(Gpu.FellBack).toBe(true);
});

test("Go FFI kernel agrees with the inline TS path within f64 accumulation tolerance", () => {
  // The Go kernel used to be BIT-identical to TS, and that property was its only correctness guard
  // (checked by hand in Scripts/ComputeSpike.ts). It is now vectorized (AVX2/FMA: 4 partial sums, and
  // FMA rounds once instead of twice), so bit-equality is gone by design — which leaves the kernel
  // otherwise UNGUARDED, because RunGradCheck never activates a backend and so only ever exercises the
  // inline TS path. This test is the replacement guard: agreement to f64 accumulation noise.
  //
  // The shapes deliberately hit the kernel's edge paths, which is where a blocked+vectorized kernel
  // actually breaks: M % 8 != 0 (the one-row-at-a-time remainder loop) and K % 4 != 0 (the scalar tail
  // after the vector body), plus both at once, plus a large aligned shape for the main path.
  const Ffi = ActivateFromConfig(LoadConfig({ Overrides: { Compute: { Backend: "GoFfi", Precision: "F64" } }, UseCli: false, UseEnv: false }));
  if (Ffi.FellBack) return; // no DLL on this machine (e.g. CI) — nothing to check
  const Backend = GetActiveBackend();
  if (Backend === null) throw new Error("GoFfi reported active but GetActiveBackend() is null");

  const Rng = new SeededRng(9);
  const Shapes: [number, number, number][] = [
    [256, 128, 512], // main blocked path, K % 4 == 0, M % 8 == 0
    [7, 5, 3], // both edges: M % 8 != 0 AND K % 4 != 0
    [13, 128, 9], // M % 8 != 0 only
    [16, 13, 5], // K % 4 != 0 only
    [1, 1, 1], // degenerate
  ];
  for (const [M, K, N] of Shapes) {
    const A = new Float64Array(M * K);
    const B = new Float64Array(K * N);
    for (let I = 0; I < A.length; I++) A[I] = Rng.NextGaussian();
    for (let I = 0; I < B.length; I++) B[I] = Rng.NextGaussian();
    const Go = Backend.MatMul(A, B, M, K, N);
    const Ref = new TsBackend().MatMul(A, B, M, K, N);
    let MaxRel = 0;
    for (let I = 0; I < Ref.length; I++) {
      MaxRel = Math.max(MaxRel, Math.abs(Ref[I] - Go[I]) / (Math.abs(Ref[I]) + 1e-9));
    }
    expect(MaxRel).toBeLessThan(1e-10); // reordered f64 sums drift ~1e-14; 1e-10 catches real bugs
  }
});

test("GoFfi Into/NtAcc/TnAcc kernels match the TS reference and honor accumulate semantics", () => {
  // These are the backward-plumbing entry points: MatMulInto (zero-copy forward), MatMulNtAcc
  // (dA += dOut @ Bᵀ with B consumed in place) and MatMulTnAcc (dB += Aᵀ @ dOut with A consumed in
  // place). They accumulate into SEEDED buffers here because that is their real calling convention —
  // grad buffers already hold earlier sequences' gradients. Shapes hit the same kernel edge paths as
  // the forward test (M % 8, K % 4, degenerate).
  const Ffi = ActivateFromConfig(LoadConfig({ Overrides: { Compute: { Backend: "GoFfi", Precision: "F64" } }, UseCli: false, UseEnv: false }));
  if (Ffi.FellBack) return; // no DLL on this machine (e.g. CI) — nothing to check
  const Backend = GetActiveBackend();
  if (Backend === null) throw new Error("GoFfi reported active but GetActiveBackend() is null");
  if (Backend.MatMulInto === undefined || Backend.MatMulNtAcc === undefined || Backend.MatMulTnAcc === undefined) {
    // A DLL predating the backward kernels would silently fall back to the slow JS transpose path —
    // that is graceful in production but in THIS repo it means someone forgot to rebuild.
    throw new Error("GoKernels/matmul.dll is stale: NT/TN symbols missing — rebuild via `go build -buildmode=c-shared -o ../matmul.dll ./` in GoKernels/ffi (PowerShell)");
  }

  const Transpose = (Data: Float64Array, Rows: number, Cols: number): Float64Array => {
    const T = new Float64Array(Rows * Cols);
    for (let R = 0; R < Rows; R++) for (let C = 0; C < Cols; C++) T[C * Rows + R] = Data[R * Cols + C];
    return T;
  };
  const FillRand = (Size: number, Rng: SeededRng): Float64Array => {
    const Buf = new Float64Array(Size);
    for (let I = 0; I < Size; I++) Buf[I] = Rng.NextGaussian();
    return Buf;
  };
  // Error relative to max(|Want|, 1): the accumulate variants ADD a random seed to the product, so
  // an element can cancel to ~1e-4 while carrying the full ~1e-14 absolute rounding noise of a
  // 256-term sum — a pure-relative metric then reports ~1e-10 for a correct kernel (observed). A
  // real kernel bug (wrong index, missed tail, overwrite-vs-accumulate) is an O(1) error either way.
  const ExpectClose = (Got: Float64Array, Want: Float64Array): void => {
    let MaxRel = 0;
    for (let I = 0; I < Want.length; I++) {
      MaxRel = Math.max(MaxRel, Math.abs(Want[I] - Got[I]) / Math.max(Math.abs(Want[I]), 1));
    }
    expect(MaxRel).toBeLessThan(1e-10);
  };

  const Ref = new TsBackend();
  const Rng = new SeededRng(11);
  const Shapes: [number, number, number][] = [
    [256, 128, 512], // main blocked path
    [7, 5, 3], // M % 8 != 0 AND K % 4 != 0
    [13, 128, 9], // M % 8 != 0 only
    [16, 13, 5], // K % 4 != 0 only
    [1, 1, 1], // degenerate
  ];
  for (const [M, K, N] of Shapes) {
    const A = FillRand(M * K, Rng);
    const B = FillRand(K * N, Rng);

    // MatMulInto: overwrites a dirty buffer completely.
    const Into = new Float64Array(M * N).fill(7);
    Backend.MatMulInto(A, B, Into, M, K, N);
    ExpectClose(Into, Ref.MatMul(A, B, M, K, N));

    // MatMulNtAcc: Out[M,N] += A[M,K] @ Btᵀ with Bt stored [N,K].
    const Bt = FillRand(N * K, Rng);
    const Seed = FillRand(M * N, Rng);
    const NtOut = Float64Array.from(Seed);
    Backend.MatMulNtAcc(A, Bt, NtOut, M, K, N);
    const NtWant = Ref.MatMul(A, Transpose(Bt, N, K), M, K, N);
    for (let I = 0; I < NtWant.length; I++) NtWant[I] += Seed[I];
    ExpectClose(NtOut, NtWant);

    // MatMulTnAcc: Db[K,N] += Aᵀ @ DOut with A stored [M,K].
    const DOut = FillRand(M * N, Rng);
    const DbSeed = FillRand(K * N, Rng);
    const Db = Float64Array.from(DbSeed);
    Backend.MatMulTnAcc(A, DOut, Db, M, K, N);
    const TnWant = Ref.MatMul(Transpose(A, M, K), DOut, K, M, N);
    for (let I = 0; I < TnWant.length; I++) TnWant[I] += DbSeed[I];
    ExpectClose(Db, TnWant);
  }
});

test("GoFfi-routed backward gradients match the inline backward within kernel tolerance", () => {
  // End-to-end guard for the Ops/MatMul wiring of the native backward halves: the same tape node,
  // once through the inline scalar path and once through GoFfi with NtAcc/TnAcc, must produce the
  // same gradients up to the kernel's known reordered-sum/FMA drift. Odd shapes on purpose (M % 8,
  // K % 4) so the remainder paths are the ones wired in.
  const Ffi = ActivateFromConfig(LoadConfig({ Overrides: { Compute: { Backend: "GoFfi", Precision: "F64" } }, UseCli: false, UseEnv: false }));
  if (Ffi.FellBack) return; // no DLL on this machine (e.g. CI) — nothing to check

  const Rng = new SeededRng(12);
  const A = RandTensor(13, 7, Rng);
  const B = RandTensor(7, 9, Rng);
  const UpstreamGrad = new Float64Array(13 * 9);
  for (let I = 0; I < UpstreamGrad.length; I++) UpstreamGrad[I] = Rng.NextGaussian();

  SetActiveBackend(null);
  const InlineOut = MatMul(A, B);
  InlineOut.Grad.set(UpstreamGrad);
  InlineOut.BackwardFn?.();
  const InlineDA = Float64Array.from(A.Grad);
  const InlineDB = Float64Array.from(B.Grad);

  A.Grad.fill(0);
  B.Grad.fill(0);
  ActivateFromConfig(LoadConfig({ Overrides: { Compute: { Backend: "GoFfi", Precision: "F64" } }, UseCli: false, UseEnv: false }));
  const RoutedOut = MatMul(A, B);
  RoutedOut.Grad.set(UpstreamGrad);
  RoutedOut.BackwardFn?.();
  for (let I = 0; I < A.Size; I++) {
    expect(Math.abs(A.Grad[I] - InlineDA[I]) / (Math.abs(InlineDA[I]) + 1e-9)).toBeLessThan(1e-10);
  }
  for (let I = 0; I < B.Size; I++) {
    expect(Math.abs(B.Grad[I] - InlineDB[I]) / (Math.abs(InlineDB[I]) + 1e-9)).toBeLessThan(1e-10);
  }
});

test("switching away from the Go FFI backend leaves an already-captured backend callable", () => {
  // Ops/MatMul stores the ACTIVE backend in every tape node's backward closure, so a node built while
  // GoFfi was active still calls into the DLL after a switch (Backward runs later). The selector used
  // to Close() the outgoing handle, unloading the library under those closures: the next call jumped
  // into freed code and took the whole process down with a SIGSEGV. A segfault is not throwable, so
  // `expect().not.toThrow()` could never catch it — reaching the assertion at all IS the proof.
  const Ffi = ActivateFromConfig(LoadConfig({ Overrides: { Compute: { Backend: "GoFfi", Precision: "F64" } }, UseCli: false, UseEnv: false }));
  if (Ffi.FellBack) return; // no DLL on this machine (e.g. CI) — nothing to regress
  const Captured = GetActiveBackend();
  if (Captured === null) throw new Error("GoFfi reported active but GetActiveBackend() is null");

  SetActiveBackend(null); // the switch that used to unload the DLL

  const Out = Captured.MatMul(new Float64Array([1, 2, 3, 4]), new Float64Array([1, 0, 0, 1]), 2, 2, 2);
  expect(Array.from(Out)).toEqual([1, 2, 3, 4]); // identity multiply still correct after the switch
});

test("MatMulNtInto (Out = A @ Btᵀ, overwrite) agrees across TsBackend, TsBackendF32 and GoFfi", () => {
  // The NT forward entry used by attention scores / tied heads: Bt arrives stored [N,K], and Out is
  // fully OVERWRITTEN (dirty buffers prove no accumulation leaks in). Reference = MatMul(A, Btᵀ).
  const Rng = new SeededRng(31);
  const Shapes: [number, number, number][] = [[33, 70, 19], [7, 5, 3], [1, 1, 1]];
  const Ts = new TsBackend();
  const Ffi = ActivateFromConfig(LoadConfig({ Overrides: { Compute: { Backend: "GoFfi", Precision: "F64" } }, UseCli: false, UseEnv: false }));
  const Native = Ffi.FellBack ? null : GetActiveBackend();
  SetActiveBackend(null);
  for (const [M, K, N] of Shapes) {
    const A = new Float64Array(M * K);
    const Bt = new Float64Array(N * K);
    for (let I = 0; I < A.length; I++) A[I] = Rng.NextGaussian();
    for (let I = 0; I < Bt.length; I++) Bt[I] = Rng.NextGaussian();
    const B = new Float64Array(K * N);
    for (let R = 0; R < N; R++) for (let C = 0; C < K; C++) B[C * N + R] = Bt[R * K + C];
    const Want = Ts.MatMul(A, B, M, K, N);

    const ScaledErr = (Got: ArrayLike<number>): number => {
      let Max = 0;
      for (let I = 0; I < Want.length; I++) Max = Math.max(Max, Math.abs(Want[I] - Got[I]) / Math.max(Math.abs(Want[I]), 1));
      return Max;
    };
    const TsOut = new Float64Array(M * N).fill(7);
    Ts.MatMulNtInto(A, Bt, TsOut, M, K, N);
    expect(ScaledErr(TsOut)).toBeLessThan(1e-12);

    const F32Out = new Float64Array(M * N).fill(7);
    new TsBackendF32().MatMulNtInto(A, Bt, F32Out, M, K, N);
    expect(ScaledErr(F32Out)).toBeLessThan(1e-4);

    if (Native !== null) {
      if (Native.MatMulNtInto === undefined) throw new Error("GoKernels/matmul.dll is stale: rebuild it");
      const GoOut = new Float64Array(M * N).fill(7);
      Native.MatMulNtInto(A, Bt, GoOut, M, K, N);
      expect(ScaledErr(GoOut)).toBeLessThan(1e-10);
    }
  }
});
