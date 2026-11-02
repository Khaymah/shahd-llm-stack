// In-process Go compute backend via bun:ffi (ADR-0002, the FAST owned path). Loads the cgo
// c-shared DLL and calls the matmul kernels synchronously with zero serialization — so unlike the
// subprocess GoBackend it IS a drop-in sync ComputeBackend. Requires GoKernels/matmul.dll (built
// from GoKernels/ffi via `go build -buildmode=c-shared`, done in PowerShell where MinGW gcc works).
//
// FINDING: a naive scalar Go port LOST to Bun's JIT, and even the tuned scalar Go kernel was ~1.4-1.9x
// slower per core than the JIT (measured at GOMAXPROCS=1) — Go's only edge was cheap goroutine fan-out,
// because Go has no SIMD intrinsics. The kernel's inner loop is therefore C (AVX2+FMA, see
// GoKernels/ffi/matmul_avx.c) with Go keeping the row fan-out: measured 3.0-4.9x over the old scalar
// kernel across this model's real shapes, ~1.95x on a whole training step. It stays an OPT-IN
// accelerator, not the default: it needs the prebuilt DLL (gcc, built via PowerShell), so TsBackend
// remains the zero-dependency fallback the hot path can always use.
//
// Besides the allocating MatMul, this backend implements the OPTIONAL ComputeBackend methods:
// MatMulInto (writes into the caller's buffer — kills one full-copy + one allocation per forward)
// and MatMulNtAcc/MatMulTnAcc (the two backward halves as native kernels — kills the JS-side
// single-threaded transposes and scalar `+=` grad loops). The NT/TN symbols only exist in DLLs
// built after they were added, so loading falls back to the MatMulF64-only symbol set and simply
// leaves the optional methods undefined — Ops/MatMul then uses its transpose path unchanged.
//
// NOT bit-identical to TsBackend any more (vectorizing the k loop keeps 4 partial sums and FMA rounds
// once, not twice; drift is ~1e-14 relative). That was a deliberate trade for the SIMD win. The guard
// is Tests/Compute.Test.ts, which checks this kernel against the TS reference within f64 accumulation
// tolerance on shapes that hit its edge paths — RunGradCheck never activates a backend, so it only
// ever exercises the inline TS path and does NOT cover this code.
//
// SAFETY (two native hazards, both uncatchable from TS once they fire):
//   1. ISA — the kernels dispatch at load time between AVX2+FMA and a scalar copy (see
//      GoKernels/ffi/matmul_dispatch.c), and ShahdCpuSupportsAvx2Fma reports which one runs.
//      CpuSupportsAvx2Fma exposes that to BackendSelector; null means the library predates the
//      probe (and the dispatcher), i.e. it may SIGILL on a CPU without AVX2/FMA.
//   2. Memory — every entry validates its buffers first (ValidateMatMulArgs: one element type,
//      lengths >= the shape's products, nothing empty) so a short or mixed-width buffer throws a
//      descriptive Error instead of being read/written out of bounds in C.
//
// The DLL is opened ONCE per path and kept for the PROCESS LIFETIME — never unloaded. That is a
// CORRECTNESS requirement, not an optimization: Ops/MatMul captures the active backend inside every
// tape node's backward closure, so a node built while this backend was active still calls into the
// library long after a SetActiveBackend switch. Unloading left those closures jumping into freed code
// and killed the process with a SIGSEGV (unrecoverable — no catchable error, no stack). Caching the
// handle also makes repeated ActivateFromConfig calls leak nothing, which is what the old
// close-on-switch was there for.

import { dlopen, FFIType, ptr } from "bun:ffi";
import type { Pointer } from "bun:ffi";
import type { ComputeBackend } from "./ComputeBackend.ts";
import type { NumArray, NumArrayCtor } from "../Tensor/Tensor.ts";
import { ValidateMatMulArgs } from "./MatMulArgs.ts";

type MatMulSymbol = (a: Pointer, b: Pointer, out: Pointer, m: number, k: number, n: number) => void;
type MatMulAccSymbol = (a: Pointer, b: Pointer, out: Pointer, m: number, k: number, n: number, acc: number) => void;
type IntArgSymbol = (n: number) => void;
type ProbeSymbol = () => number;
type KernelSymbols = {
  MatMul: MatMulSymbol;
  Nt: MatMulAccSymbol | null;
  Tn: MatMulAccSymbol | null;
  // The f32 trio (8 SIMD lanes instead of 4). Null on DLLs built before the f32 kernels existed;
  // BackendSelector treats a null here as "GoFfi cannot serve an F32 run" and falls back to CPU.
  MatMul32: MatMulSymbol | null;
  Nt32: MatMulAccSymbol | null;
  Tn32: MatMulAccSymbol | null;
  Threads: IntArgSymbol | null;
  // CPU-feature probe + scalar-forcing test hook, added together with the ISA dispatcher. Null on
  // older DLLs — which compile every kernel as AVX2 with no runtime check (SIGILL risk).
  Probe: ProbeSymbol | null;
  ForceScalar: IntArgSymbol | null;
};

const DefaultLibPath = "GoKernels/matmul.dll";

const MatMulDef = {
  args: [FFIType.ptr, FFIType.ptr, FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.i32],
  returns: FFIType.void,
} as const;
const MatMulAccDef = {
  args: [FFIType.ptr, FFIType.ptr, FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.i32, FFIType.i32],
  returns: FFIType.void,
} as const;
const IntArgDef = { args: [FFIType.i32], returns: FFIType.void } as const;
const ProbeDef = { args: [], returns: FFIType.i32 } as const;

// Accumulate flag values for the NT/TN kernels (C `int Acc`).
const Overwrite = 0;
const Accumulate = 1;

// Strong, process-lifetime refs keyed by library path (see the header for why these are never closed).
const LoadedLibs = new Map<string, KernelSymbols>();

type CoreSymbols = Omit<KernelSymbols, "Probe" | "ForceScalar">;

function LoadCoreSymbols(LibPath: string): CoreSymbols {
  try {
    const Lib = dlopen(LibPath, {
      MatMulF64: MatMulDef,
      MatMulNTF64: MatMulAccDef,
      MatMulTNF64: MatMulAccDef,
      MatMulF32: MatMulDef,
      MatMulNTF32: MatMulAccDef,
      MatMulTNF32: MatMulAccDef,
      SetKernelThreads: IntArgDef,
    });
    return {
      MatMul: Lib.symbols.MatMulF64 as unknown as MatMulSymbol,
      Nt: Lib.symbols.MatMulNTF64 as unknown as MatMulAccSymbol,
      Tn: Lib.symbols.MatMulTNF64 as unknown as MatMulAccSymbol,
      MatMul32: Lib.symbols.MatMulF32 as unknown as MatMulSymbol,
      Nt32: Lib.symbols.MatMulNTF32 as unknown as MatMulAccSymbol,
      Tn32: Lib.symbols.MatMulTNF32 as unknown as MatMulAccSymbol,
      Threads: Lib.symbols.SetKernelThreads as unknown as IntArgSymbol,
    };
  } catch {
    try {
      // DLL predating the f32 kernels — full f64 set only.
      const Lib = dlopen(LibPath, {
        MatMulF64: MatMulDef,
        MatMulNTF64: MatMulAccDef,
        MatMulTNF64: MatMulAccDef,
        SetKernelThreads: IntArgDef,
      });
      return {
        MatMul: Lib.symbols.MatMulF64 as unknown as MatMulSymbol,
        Nt: Lib.symbols.MatMulNTF64 as unknown as MatMulAccSymbol,
        Tn: Lib.symbols.MatMulTNF64 as unknown as MatMulAccSymbol,
        MatMul32: null,
        Nt32: null,
        Tn32: null,
        Threads: Lib.symbols.SetKernelThreads as unknown as IntArgSymbol,
      };
    } catch {
      // Oldest DLL without the backward entry points (or the throw is "no such file" — then this
      // dlopen throws too, which is the constructor's absent-DLL signal for TryGoFfi).
      const Lib = dlopen(LibPath, { MatMulF64: MatMulDef });
      return { MatMul: Lib.symbols.MatMulF64 as unknown as MatMulSymbol, Nt: null, Tn: null, MatMul32: null, Nt32: null, Tn32: null, Threads: null };
    }
  }
}

// The probe lives in its own dlopen so a DLL missing it still loads (with Probe = null) instead of
// failing the whole core symbol set — absence is information the selector acts on.
function LoadProbeSymbols(LibPath: string): Pick<KernelSymbols, "Probe" | "ForceScalar"> {
  try {
    const Lib = dlopen(LibPath, { ShahdCpuSupportsAvx2Fma: ProbeDef, SetScalarKernels: IntArgDef });
    return {
      Probe: Lib.symbols.ShahdCpuSupportsAvx2Fma as unknown as ProbeSymbol,
      ForceScalar: Lib.symbols.SetScalarKernels as unknown as IntArgSymbol,
    };
  } catch {
    return { Probe: null, ForceScalar: null };
  }
}

function LoadKernelSymbols(LibPath: string): KernelSymbols {
  const Cached = LoadedLibs.get(LibPath);
  if (Cached !== undefined) return Cached;
  const Symbols: KernelSymbols = { ...LoadCoreSymbols(LibPath), ...LoadProbeSymbols(LibPath) };
  LoadedLibs.set(LibPath, Symbols);
  return Symbols;
}

// bun:ffi's ptr() does not throw on a zero-length view in every version — it can hand back an Error
// object instead of a number, which would then be passed to C as a "pointer". ValidateMatMulArgs
// already rejects empty buffers; this is the last line of defense for anything it did not foresee.
function BufPtr(Buf: NumArray): Pointer {
  const P: unknown = ptr(Buf);
  if (typeof P !== "number" || P === 0) {
    throw new Error(`GoFfiBackend: could not take a native pointer to a ${Buf.length}-element buffer`);
  }
  return P as Pointer;
}

function Rebuild(What: string): Error {
  return new Error(`GoFfiBackend: DLL lacks the ${What} — rebuild GoKernels (cd GoKernels/ffi && go build -buildmode=c-shared -o ../matmul.dll ./)`);
}

export class GoFfiBackend implements ComputeBackend {
  private Symbols: KernelSymbols;

  /** True when the DLL exports the f32 kernel trio — BackendSelector requires this for F32 runs. */
  readonly HasF32: boolean;

  /** true: AVX2+FMA kernels run. false: this CPU only gets the scalar fallback. null: the library
   *  predates the CPU probe — its kernels are AVX2-only with no runtime check, so calling them on a
   *  CPU without AVX2/FMA is a SIGILL. BackendSelector only activates this backend on `true`. */
  readonly CpuSupportsAvx2Fma: boolean | null;

  // Optional ComputeBackend capabilities — only assigned when the DLL exports the new symbols, so
  // `Backend.MatMulNtAcc === undefined` remains an honest feature probe for Ops/MatMul.
  MatMulNtAcc?: (A: NumArray, Bt: NumArray, Out: NumArray, M: number, K: number, N: number) => void;
  MatMulNtInto?: (A: NumArray, Bt: NumArray, Out: NumArray, M: number, K: number, N: number) => void;
  MatMulTnAcc?: (A: NumArray, DOut: NumArray, Db: NumArray, M: number, K: number, N: number) => void;

  /** PROCESS-GLOBAL cap on per-call goroutine fan-out (one Go runtime serves every JS thread).
   *  The training worker pool sets 1 around its dispatch — its JS workers ARE the parallelism —
   *  and 0 restores the all-cores default. Only defined when the DLL exports the symbol. */
  SetKernelThreads?: (N: number) => void;

  /** PROCESS-GLOBAL test hook: true forces the scalar kernels, false restores CPU-based dispatch
   *  (never enables AVX on a CPU without it). Only defined when the DLL has the ISA dispatcher. */
  SetScalarKernels?: (Force: boolean) => void;

  constructor(LibPath = DefaultLibPath) {
    this.Symbols = LoadKernelSymbols(LibPath); // throws when the DLL is absent — TryGoFfi probes with this
    const { Nt, Tn, Nt32, Tn32, Threads, Probe, ForceScalar } = this.Symbols;
    this.HasF32 = this.Symbols.MatMul32 !== null;
    this.CpuSupportsAvx2Fma = Probe === null ? null : Probe() !== 0; // the probe itself is scalar C — safe anywhere
    if (Nt !== null) {
      // Dispatch by element type per call; ValidateMatMulArgs guarantees every buffer shares it.
      const RunNt = (A: NumArray, Bt: NumArray, Out: NumArray, M: number, K: number, N: number, Acc: number, Op: string): void => {
        ValidateMatMulArgs(Op, A, Bt, Out, M, K, N, "NT");
        const Fn = A instanceof Float32Array ? Nt32 : Nt;
        if (Fn === null) throw Rebuild("f32 NT kernel");
        Fn(BufPtr(A), BufPtr(Bt), BufPtr(Out), M, K, N, Acc);
      };
      this.MatMulNtAcc = (A, Bt, Out, M, K, N) => RunNt(A, Bt, Out, M, K, N, Accumulate, "GoFfiBackend.MatMulNtAcc");
      this.MatMulNtInto = (A, Bt, Out, M, K, N) => RunNt(A, Bt, Out, M, K, N, Overwrite, "GoFfiBackend.MatMulNtInto");
    }
    if (Tn !== null) {
      this.MatMulTnAcc = (A, DOut, Db, M, K, N) => {
        ValidateMatMulArgs("GoFfiBackend.MatMulTnAcc", A, DOut, Db, M, K, N, "TN");
        const Fn = A instanceof Float32Array ? Tn32 : Tn;
        if (Fn === null) throw Rebuild("f32 TN kernel");
        Fn(BufPtr(A), BufPtr(DOut), BufPtr(Db), M, K, N, Accumulate);
      };
    }
    if (Threads !== null) {
      this.SetKernelThreads = (N) => Threads(N);
    }
    if (ForceScalar !== null) {
      this.SetScalarKernels = (Force) => ForceScalar(Force ? 1 : 0);
    }
  }

  MatMul(A: NumArray, B: NumArray, M: number, K: number, N: number): NumArray {
    const Out = new (A.constructor as NumArrayCtor)(M * N);
    this.MatMulInto(A, B, Out, M, K, N);
    return Out;
  }

  MatMulInto(A: NumArray, B: NumArray, Out: NumArray, M: number, K: number, N: number): void {
    ValidateMatMulArgs("GoFfiBackend.MatMulInto", A, B, Out, M, K, N, "NN");
    const Fn = A instanceof Float32Array ? this.Symbols.MatMul32 : this.Symbols.MatMul;
    if (Fn === null) throw Rebuild("f32 forward kernel");
    Fn(BufPtr(A), BufPtr(B), BufPtr(Out), M, K, N); // kernel fully overwrites Out
  }
}
