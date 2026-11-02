import { test, expect, afterEach } from "bun:test";
import {
  ActivateFromConfig,
  GetActiveBackend,
  SetActiveBackend,
} from "../Brain/ComputeBackend/BackendSelector.ts";
import type { GoFfiCandidate, SelectorDeps } from "../Brain/ComputeBackend/BackendSelector.ts";
import { GoFfiBackend } from "../Brain/ComputeBackend/GoFfiBackend.ts";
import { ValidateMatMulArgs } from "../Brain/ComputeBackend/MatMulArgs.ts";
import { TsBackend } from "../Brain/ComputeBackend/TsBackend.ts";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { SeededRng } from "../Brain/Random/SeededRng.ts";
import type { NumArray, NumArrayCtor } from "../Brain/Tensor/Tensor.ts";

// Never leave a backend active (or F32 storage selected) for the gradcheck-bearing tests elsewhere.
afterEach(() => {
  ActivateFromConfig(LoadConfig({ Overrides: { Compute: { Backend: "Ts", Precision: "F64" } }, UseCli: false, UseEnv: false }));
  SetActiveBackend(null);
});

function GoFfiConfig(FallbackToCpu: boolean) {
  return LoadConfig({ Overrides: { Compute: { Backend: "GoFfi", Precision: "F64", FallbackToCpu } }, UseCli: false, UseEnv: false });
}

/** A stand-in for the loaded library whose CPU probe answers `Probe` — no DLL needed. Its MatMul
 *  throws so a test fails loudly if the selector ever routes work to a library it should reject. */
function FakeLibrary(Probe: boolean | null): GoFfiCandidate {
  return {
    HasF32: true,
    CpuSupportsAvx2Fma: Probe,
    MatMul: () => {
      throw new Error("fake library must never be called");
    },
  };
}

function Deps(Probe: boolean | null, Warnings: string[]): SelectorDeps {
  return { LoadGoFfi: () => FakeLibrary(Probe), Warn: (Message) => Warnings.push(Message) };
}

// ── CPU guard ────────────────────────────────────────────────────────────────────────────────────

test("CPU probe = 0 (no AVX2/FMA): selector warns once and activates the TS backend", () => {
  const Warnings: string[] = [];
  const Choice = ActivateFromConfig(GoFfiConfig(true), Deps(false, Warnings));
  expect(Choice.FellBack).toBe(true);
  expect(Choice.Chosen).toContain("CPU fallback");
  expect(GetActiveBackend()).toBe(null); // the inline TS path
  expect(Warnings.length).toBe(1);
  expect(Warnings[0]).toContain("AVX2/FMA");
});

test("missing probe symbol (old library build) is treated as unsafe and asks for a rebuild", () => {
  const Warnings: string[] = [];
  const Choice = ActivateFromConfig(GoFfiConfig(true), Deps(null, Warnings));
  expect(Choice.FellBack).toBe(true);
  expect(GetActiveBackend()).toBe(null);
  expect(Warnings[0]).toContain("rebuild");
  expect(Choice.Reason).toContain("SIGILL");
});

test("unsafe CPU with FallbackToCpu=false throws instead of risking a SIGILL", () => {
  expect(() => ActivateFromConfig(GoFfiConfig(false), Deps(false, []))).toThrow(/AVX2\/FMA/);
  expect(GetActiveBackend()).toBe(null);
});

test("CPU probe = 1 activates the library; an absent library falls back silently", () => {
  const Lib = FakeLibrary(true);
  const Ok = ActivateFromConfig(GoFfiConfig(true), { LoadGoFfi: () => Lib, Warn: () => undefined });
  expect(Ok.FellBack).toBe(false);
  expect(GetActiveBackend()).toBe(Lib);

  const Warnings: string[] = [];
  const Absent = ActivateFromConfig(GoFfiConfig(true), {
    LoadGoFfi: () => {
      throw new Error("no such file");
    },
    Warn: (Message) => Warnings.push(Message),
  });
  expect(Absent.FellBack).toBe(true);
  expect(Warnings.length).toBe(0); // no DLL is the normal CI case, not worth a warning
});

// ── Argument validation (pure — no DLL needed) ───────────────────────────────────────────────────

test("ValidateMatMulArgs rejects short, mixed-precision, empty and malformed operands", () => {
  const F64 = (Size: number): Float64Array => new Float64Array(Size);
  const Op = "Test";
  expect(() => ValidateMatMulArgs(Op, F64(6), F64(12), F64(8), 2, 3, 4)).not.toThrow();

  expect(() => ValidateMatMulArgs(Op, F64(5), F64(12), F64(8), 2, 3, 4)).toThrow(/A has 5 elements/);
  expect(() => ValidateMatMulArgs(Op, F64(6), F64(11), F64(8), 2, 3, 4)).toThrow(/B has 11 elements/);
  expect(() => ValidateMatMulArgs(Op, F64(6), F64(12), F64(7), 2, 3, 4)).toThrow(/Out has 7 elements/);

  expect(() => ValidateMatMulArgs(Op, F64(6), new Float32Array(12), F64(8), 2, 3, 4)).toThrow(/mixed buffer precisions/);
  expect(() => ValidateMatMulArgs(Op, new Float32Array(6), new Float32Array(12), F64(8), 2, 3, 4)).toThrow(/mixed/);

  expect(() => ValidateMatMulArgs(Op, F64(0), F64(12), F64(8), 2, 3, 4)).toThrow(/empty/);
  expect(() => ValidateMatMulArgs(Op, F64(0), F64(0), F64(0), 0, 0, 0)).toThrow(/dimension M/);
  expect(() => ValidateMatMulArgs(Op, F64(6), F64(12), F64(8), 2, 1.5, 4)).toThrow(/dimension K/);
  expect(() => ValidateMatMulArgs(Op, F64(6), F64(12), F64(8), 2, 3, Number.NaN)).toThrow(/dimension N/);
  const NotTyped = [1, 2, 3, 4, 5, 6] as unknown as NumArray;
  expect(() => ValidateMatMulArgs(Op, NotTyped, F64(12), F64(8), 2, 3, 4)).toThrow(/Float64Array or Float32Array/);

  // TN: A[M,K], DOut[M,N], Db[K,N] — DOut is sized by M*N, Db by K*N.
  expect(() => ValidateMatMulArgs(Op, F64(6), F64(8), F64(12), 2, 3, 4, "TN")).not.toThrow();
  expect(() => ValidateMatMulArgs(Op, F64(6), F64(8), F64(11), 2, 3, 4, "TN")).toThrow(/Db has 11 elements/);
  expect(() => ValidateMatMulArgs(Op, F64(6), F64(7), F64(12), 2, 3, 4, "TN")).toThrow(/DOut has 7 elements/);
});

// ── Real library (skipped when GoKernels/matmul.dll is not built on this machine) ────────────────

function TryRealLibrary(): GoFfiBackend | null {
  try {
    return new GoFfiBackend();
  } catch {
    return null;
  }
}

test("every GoFfiBackend entry point validates before touching native memory", () => {
  const Lib = TryRealLibrary();
  if (Lib === null) return; // no DLL on this machine — the pure validation test above still ran
  const Short = new Float64Array(3);
  const Ok = (Size: number): Float64Array => new Float64Array(Size);
  expect(() => Lib.MatMulInto(Short, Ok(12), Ok(8), 2, 3, 4)).toThrow(/MatMulInto/);
  expect(() => Lib.MatMul(Ok(6), new Float32Array(12), 2, 3, 4)).toThrow(/mixed/);
  expect(() => Lib.MatMulNtAcc?.(Ok(6), Ok(12), new Float64Array(0), 2, 3, 4)).toThrow(/empty/);
  expect(() => Lib.MatMulNtInto?.(Ok(6), Short, Ok(8), 2, 3, 4)).toThrow(/MatMulNtInto/);
  expect(() => Lib.MatMulTnAcc?.(Ok(6), Ok(8), Short, 2, 3, 4)).toThrow(/MatMulTnAcc/);
  expect(() => Lib.MatMulInto(Ok(1), Ok(1), Ok(1), 0, 1, 1)).toThrow(/dimension M/);
});

function Transpose(Data: Float64Array, Rows: number, Cols: number): Float64Array {
  const Out = new Float64Array(Rows * Cols);
  for (let R = 0; R < Rows; R++) for (let C = 0; C < Cols; C++) Out[C * Rows + R] = Data[R * Cols + C];
  return Out;
}

function MaxScaledError(Got: NumArray, Want: Float64Array): number {
  let Max = 0;
  for (let I = 0; I < Want.length; I++) Max = Math.max(Max, Math.abs(Want[I] - Got[I]) / Math.max(Math.abs(Want[I]), 1));
  return Max;
}

test("forced scalar kernels (the no-AVX2 fallback) match the TS reference in f64 and f32", () => {
  const Lib = TryRealLibrary();
  if (Lib === null || Lib.SetScalarKernels === undefined) return; // no DLL, or one without the dispatcher
  const NtAcc = Lib.MatMulNtAcc;
  const NtInto = Lib.MatMulNtInto;
  const TnAcc = Lib.MatMulTnAcc;
  if (NtAcc === undefined || NtInto === undefined || TnAcc === undefined) throw new Error("stale DLL: rebuild GoKernels");

  const Ref = new TsBackend();
  const Rng = new SeededRng(21);
  const Cases: { Ctor: NumArrayCtor; Tolerance: number }[] = [
    { Ctor: Float64Array, Tolerance: 1e-10 },
    { Ctor: Float32Array, Tolerance: 1e-4 },
  ];
  // Edge shapes of the AVX kernels (M % 8, K % 4/8, N tails) — the scalar copy must agree on all.
  const Shapes: [number, number, number][] = [[33, 70, 19], [7, 5, 3], [1, 1, 1]];
  Lib.SetScalarKernels(true);
  try {
    for (const { Ctor, Tolerance } of Cases) {
      for (const [M, K, N] of Shapes) {
        const Rand = (Size: number): Float64Array => {
          const Buf = new Float64Array(Size);
          for (let I = 0; I < Size; I++) Buf[I] = Rng.NextGaussian();
          return Ctor === Float32Array ? Float64Array.from(Float32Array.from(Buf)) : Buf; // f32-exact inputs
        };
        const A = Rand(M * K);
        const B = Rand(K * N);
        const Bt = Rand(N * K);
        const DOut = Rand(M * N);
        const Seed = Rand(M * N);
        const DbSeed = Rand(K * N);

        const Into = new Ctor(M * N).fill(7);
        Lib.MatMulInto(Ctor.from(A), Ctor.from(B), Into, M, K, N);
        expect(MaxScaledError(Into, Ref.MatMul(A, B, M, K, N))).toBeLessThan(Tolerance);

        const NtWant = Ref.MatMul(A, Transpose(Bt, N, K), M, K, N);
        const NtOut = new Ctor(M * N).fill(7);
        NtInto(Ctor.from(A), Ctor.from(Bt), NtOut, M, K, N);
        expect(MaxScaledError(NtOut, NtWant)).toBeLessThan(Tolerance);

        const NtAccOut = Ctor.from(Seed);
        NtAcc(Ctor.from(A), Ctor.from(Bt), NtAccOut, M, K, N);
        expect(MaxScaledError(NtAccOut, NtWant.map((V, I) => V + Seed[I]))).toBeLessThan(Tolerance);

        const Db = Ctor.from(DbSeed);
        TnAcc(Ctor.from(A), Ctor.from(DOut), Db, M, K, N);
        const TnWant = Ref.MatMul(Transpose(A, M, K), DOut, K, M, N).map((V, I) => V + DbSeed[I]);
        expect(MaxScaledError(Db, TnWant)).toBeLessThan(Tolerance);
      }
    }
  } finally {
    Lib.SetScalarKernels(false); // process-global: restore CPU-based dispatch for every other test
  }
});
