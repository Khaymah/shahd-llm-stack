// The runtime-switchable compute backend selector (M2). `Ops/MatMul` asks `GetActiveBackend()` which
// backend to route through; **null means the inline f64 fast path** (the default — zero seam
// overhead, bit-identical, gradcheck-exact). `ActivateFromConfig` maps `Config.Compute` to a backend,
// probing availability (the Go FFI DLL) and falling back to CPU when `FallbackToCpu` is set, so a
// machine without a working GPU/FFI still runs. The active backend can be swapped at ANY time
// (e.g. turn a GPU off → CPU) via `SetActiveBackend`.
//
// CPU GUARD: a successful dlopen is NOT proof the native kernels can run. Libraries built before
// the ISA dispatcher compile every kernel as AVX2+FMA with no CPUID check, and on a CPU without
// those extensions the first matmul raises SIGILL — fatal, uncatchable, so FallbackToCpu would
// never run. TryGoFfi therefore asks the library's ShahdCpuSupportsAvx2Fma probe right after
// loading it and only accepts GoFfi when the answer is a definite yes: "no" (the library would only
// run its scalar fallback — the TS path is the better-tested reference there) and "unknown" (an old
// library without the probe — unsafe) both fall back with a one-line warning.

import type { ResolvedConfig } from "../Config/ConfigTypes.ts";
import type { ComputeBackend } from "./ComputeBackend.ts";
import { SetTensorPrecision } from "../Tensor/Tensor.ts";
import { GoFfiBackend } from "./GoFfiBackend.ts";

let Active: ComputeBackend | null = null; // null => Ops/MatMul uses its inline f64 fast path

/** The backend Ops routes through, or null for the inline f64 fast path. */
export function GetActiveBackend(): ComputeBackend | null {
  return Active;
}

/** Swap the active backend at runtime; pass null to return to the inline CPU f64 path. Switching only
 *  changes what NEW ops route through — a backend switched AWAY from must stay callable, because
 *  Ops/MatMul captures the active backend in the backward closure of every tape node built while it
 *  was active, and those closures run later (during Backward). This used to Close() the outgoing Go
 *  FFI handle, which unloaded the DLL out from under exactly those closures and segfaulted the whole
 *  process. GoFfiBackend now holds one cached, process-lifetime handle per path, so a switch has
 *  nothing to free and nothing to leak. */
export function SetActiveBackend(Backend: ComputeBackend | null): void {
  Active = Backend;
}

/** What ActivateFromConfig chose. Reason explains a fallback (absent DLL, unsafe CPU, ...). */
export type BackendChoice = { Chosen: string; FellBack: boolean; Reason?: string };

/** The slice of GoFfiBackend the selector needs — an interface so tests can inject a fake library
 *  (e.g. one whose CPU probe answers "no") without a real DLL on disk. */
export type GoFfiCandidate = ComputeBackend & {
  readonly HasF32: boolean;
  readonly CpuSupportsAvx2Fma: boolean | null;
};

/** Injectable side effects of ActivateFromConfig. Production uses DefaultSelectorDeps. */
export type SelectorDeps = {
  /** Load the Go FFI library; throws when it is absent or unloadable. */
  LoadGoFfi: () => GoFfiCandidate;
  /** Emit a one-line operator warning (a fallback the user should know about). */
  Warn: (Message: string) => void;
};

const RebuildHint = "rebuild it: cd GoKernels/ffi && go build -buildmode=c-shared -o ../matmul.dll ./";

// Each distinct warning is printed once per thread: ActivateFromConfig runs per test, per worker
// and per script phase, and the same fallback reason repeated adds nothing.
const WarnedOnce = new Set<string>();

export const DefaultSelectorDeps: SelectorDeps = {
  LoadGoFfi: () => new GoFfiBackend(),
  Warn: (Message) => {
    if (WarnedOnce.has(Message)) return;
    WarnedOnce.add(Message);
    console.warn(Message);
  },
};

type FfiProbe = { Backend: GoFfiCandidate; Reason: null } | { Backend: null; Reason: string };

function TryGoFfi(Deps: SelectorDeps): FfiProbe {
  let Ffi: GoFfiCandidate;
  try {
    Ffi = Deps.LoadGoFfi();
  } catch (Err) {
    // DLL/toolchain unavailable — the common, expected case on CI; no warning needed.
    return { Backend: null, Reason: `GoFfi library not loadable (${Err instanceof Error ? Err.message : String(Err)})` };
  }
  if (Ffi.CpuSupportsAvx2Fma === null) {
    const Reason = `GoFfi: GoKernels/matmul.dll predates the CPU-feature probe and may crash (SIGILL) on CPUs without AVX2/FMA — using the TS backend; ${RebuildHint}`;
    Deps.Warn(Reason);
    return { Backend: null, Reason };
  }
  if (!Ffi.CpuSupportsAvx2Fma) {
    const Reason = "GoFfi: this CPU lacks AVX2/FMA, so the native kernels would only run their scalar fallback — using the TS backend";
    Deps.Warn(Reason);
    return { Backend: null, Reason };
  }
  return { Backend: Ffi, Reason: null };
}

/** Resolve Config.Compute to an active backend, with probe + CPU fallback. Returns what was chosen.
 *  Also sets the TENSOR STORAGE precision (SetTensorPrecision) — F32 is real storage now, not a
 *  per-call conversion backend: tensors allocate Float32Array and the inline op loops work on them
 *  unchanged (JS math is f64 and rounds on store), while GoFfi dispatches to the f32 C kernels.
 *
 *  PRECISION RULE: call this BEFORE building the model/optimizer, and do not switch precision while
 *  tensors allocated under the other precision are still in use. The selector cannot enforce this —
 *  tensors are not registered anywhere it can see (Tensor.ts owns allocation) — and the test suite
 *  legitimately alternates precisions between independent models. What IS enforced: every native
 *  kernel call validates that all its buffers share one element type and are long enough
 *  (ValidateMatMulArgs), so a stale-precision tensor reaching GoFfi throws instead of corrupting
 *  memory. */
export function ActivateFromConfig(Config: ResolvedConfig, Deps: SelectorDeps = DefaultSelectorDeps): BackendChoice {
  const C = Config.Compute;
  SetTensorPrecision(C.Precision);

  if (C.Backend === "Ts") {
    SetActiveBackend(null); // the inline loops are precision-agnostic over the stored dtype
    return { Chosen: `Ts/${C.Precision} (inline)`, FellBack: false };
  }

  if (C.Backend === "GoFfi") {
    const Probe = TryGoFfi(Deps);
    // An F32 run needs the f32 kernel trio; an older DLL without it must fall back rather than
    // feed Float32Array buffers to a kernel reading doubles.
    let Reason = Probe.Reason;
    if (Probe.Backend !== null) {
      if (C.Precision === "F64" || Probe.Backend.HasF32) {
        SetActiveBackend(Probe.Backend); // GoFfiBackend reuses one cached DLL handle, so re-activating never reloads
        return { Chosen: `GoFfi/${C.Precision}`, FellBack: false };
      }
      Reason = `GoFfi: library lacks the f32 kernels needed for an F32 run; ${RebuildHint}`;
      Deps.Warn(Reason);
    }
    if (!C.FallbackToCpu) throw new Error(`GoFfi backend unavailable and Compute.FallbackToCpu is false: ${Reason}`);
    SetActiveBackend(null);
    return { Chosen: `CPU fallback (${C.Precision})`, FellBack: true, Reason: Reason ?? undefined };
  }

  // "Gpu" — reserved for M5; not built yet.
  if (!C.FallbackToCpu) throw new Error("GPU backend not built and Compute.FallbackToCpu is false");
  SetActiveBackend(null);
  return { Chosen: `CPU fallback (${C.Precision})`, FellBack: true, Reason: "GPU backend not built yet" };
}
