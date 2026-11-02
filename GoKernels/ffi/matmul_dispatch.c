// Runtime ISA dispatch for the matmul kernels. matmul.go calls ONLY the *Kernel entry points below;
// each one forwards to the AVX2+FMA kernel (matmul_avx.c) when the CPU supports it and to the
// portable scalar copy (matmul_scalar.c) otherwise.
//
// WHY: matmul_avx.c is compiled under `#pragma GCC target("avx2,fma")` with no CPUID check of its
// own. Before this file existed, the first matmul on a CPU without AVX2/FMA executed an illegal
// instruction — SIGILL kills the whole process, uncatchable from TypeScript, so Compute.FallbackToCpu
// never got a chance to run. Now the library is safe to load AND call on any CPU, and
// ShahdCpuSupportsAvx2Fma (exported from matmul.go) lets the TS BackendSelector decide whether the
// native path is worth using at all (on a scalar-only CPU it prefers the TS backend).
//
// WHY NOT target_clones: tried; gcc rejects it for this code — the "default" clone cannot inline
// the AVX intrinsics ("inlining failed in call to always_inline ... target specific option
// mismatch"), and target_clones relies on ifunc, which MinGW (the Windows build) does not provide.
// An explicit scalar copy + one predictable branch per kernel call is portable everywhere.
//
// This file is compiled WITHOUT any target pragma, so nothing here can itself raise SIGILL.

#include <stddef.h>

#if defined(__x86_64__) || defined(__i386__)
#define SHAHD_X86 1
#else
#define SHAHD_X86 0
#endif

#if SHAHD_X86
void MatMulRowsAvx(const double *A, const double *BT, double *O, int RowStart, int RowEnd, int K, int N, int Acc);
void MatMulTnAvx(const double *A, const double *DOut, double *DB, int PStart, int PEnd, int M, int K, int N, int Acc);
void MatMulRowsAvxF32(const float *A, const float *BT, float *O, int RowStart, int RowEnd, int K, int N, int Acc);
void MatMulTnAvxF32(const float *A, const float *DOut, float *DB, int PStart, int PEnd, int M, int K, int N, int Acc);
#endif
void MatMulRowsScalar(const double *A, const double *BT, double *O, int RowStart, int RowEnd, int K, int N, int Acc);
void MatMulTnScalar(const double *A, const double *DOut, double *DB, int PStart, int PEnd, int M, int K, int N, int Acc);
void MatMulRowsScalarF32(const float *A, const float *BT, float *O, int RowStart, int RowEnd, int K, int N, int Acc);
void MatMulTnScalarF32(const float *A, const float *DOut, float *DB, int PStart, int PEnd, int M, int K, int N, int Acc);

// Raw CPU fact, computed once: 1 when the CPU AND the OS (YMM state saved via XSAVE — libgcc's
// cpuinfo checks XGETBV before reporting avx/avx2) support both AVX2 and FMA, else 0.
static int CpuAvx2Fma = 0;

// Effective dispatch choice: CpuAvx2Fma unless a caller forced the scalar path (SetScalarKernels).
// Read on every kernel call from many goroutine threads, written rarely — relaxed atomics make the
// concurrent access well-defined without any fence on the hot path.
static int UseAvx = 0;

static int DetectAvx2Fma(void) {
#if SHAHD_X86
    // Required when the check may run before libgcc's own constructor (we run in one ourselves).
    __builtin_cpu_init();
    return (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) ? 1 : 0;
#else
    return 0; // non-x86: the AVX kernels are not even compiled; scalar is the only path
#endif
}

// Runs at dlopen time, before Go or TS can call any kernel, so the hot path never has to
// lazily initialize (and never races on initialization).
__attribute__((constructor)) static void InitKernelDispatch(void) {
    CpuAvx2Fma = DetectAvx2Fma();
    __atomic_store_n(&UseAvx, CpuAvx2Fma, __ATOMIC_RELAXED);
}

/** 1 when this CPU supports AVX2+FMA (the fast kernels run), 0 when only the scalar path is safe. */
int CpuHasAvx2Fma(void) {
    return CpuAvx2Fma;
}

/** Force the scalar kernels (Force != 0) or restore CPU-based dispatch (Force == 0). Exists so the
 *  scalar fallback can be parity-tested on an AVX2 machine; it never enables AVX on a CPU without it. */
void ForceScalarKernels(int Force) {
    __atomic_store_n(&UseAvx, Force ? 0 : CpuAvx2Fma, __ATOMIC_RELAXED);
}

static inline int AvxActive(void) {
    return __atomic_load_n(&UseAvx, __ATOMIC_RELAXED);
}

void MatMulRowsKernel(const double *A, const double *BT, double *O, int RowStart, int RowEnd, int K, int N, int Acc) {
#if SHAHD_X86
    if (AvxActive()) {
        MatMulRowsAvx(A, BT, O, RowStart, RowEnd, K, N, Acc);
        return;
    }
#endif
    MatMulRowsScalar(A, BT, O, RowStart, RowEnd, K, N, Acc);
}

void MatMulTnKernel(const double *A, const double *DOut, double *DB, int PStart, int PEnd, int M, int K, int N, int Acc) {
#if SHAHD_X86
    if (AvxActive()) {
        MatMulTnAvx(A, DOut, DB, PStart, PEnd, M, K, N, Acc);
        return;
    }
#endif
    MatMulTnScalar(A, DOut, DB, PStart, PEnd, M, K, N, Acc);
}

void MatMulRowsKernelF32(const float *A, const float *BT, float *O, int RowStart, int RowEnd, int K, int N, int Acc) {
#if SHAHD_X86
    if (AvxActive()) {
        MatMulRowsAvxF32(A, BT, O, RowStart, RowEnd, K, N, Acc);
        return;
    }
#endif
    MatMulRowsScalarF32(A, BT, O, RowStart, RowEnd, K, N, Acc);
}

void MatMulTnKernelF32(const float *A, const float *DOut, float *DB, int PStart, int PEnd, int M, int K, int N, int Acc) {
#if SHAHD_X86
    if (AvxActive()) {
        MatMulTnAvxF32(A, DOut, DB, PStart, PEnd, M, K, N, Acc);
        return;
    }
#endif
    MatMulTnScalarF32(A, DOut, DB, PStart, PEnd, M, K, N, Acc);
}
