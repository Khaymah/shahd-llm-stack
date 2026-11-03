// Go cgo c-shared kernel for the IN-PROCESS FFI path (ADR-0002). Built as a DLL and dlopen'd from
// TypeScript via bun:ffi — synchronous, zero IPC, so it can be a drop-in ComputeBackend. Requires
// a working C toolchain (gcc); build via PowerShell where MinGW gcc works cleanly (Git Bash
// shadows the toolchain and breaks cgo).
//
//   Build:  go build -buildmode=c-shared -o ../matmul.dll ./   (run from GoKernels/ffi, any OS)
//   (the .dll name is kept on Linux/macOS too — dlopen does not care about the extension, and
//   GoFfiBackend's default path expects exactly GoKernels/matmul.dll)
//
// This file is the THREADING + FFI shell only; the arithmetic lives in matmul_avx.c (AVX2+FMA) and
// its portable twin matmul_scalar.c, selected per call by matmul_dispatch.c from a CPUID check made
// at load time — so the library is safe on CPUs without AVX2/FMA (no SIGILL). The split is forced by
// cgo: a file using //export may only DECLARE C functions in its preamble, never define them.
//
// Why the inner loop is C, not Go: Go has no SIMD intrinsics, so a Go kernel stays scalar (it would
// need Plan9 assembly). Measured, the old scalar Go kernel ran ~16 GFLOP/s on a CPU capable of ~500,
// and was actually SLOWER per core than Bun's JIT — its only edge was goroutine parallelism. C+AVX2
// closes that gap while Go keeps doing what it is good at here: cheap fan-out across cores.
//
// THREE ENTRY POINTS, one forward + the two halves of the matmul backward:
//   MatMulF64   out = a @ b                 (transposes b internally, overwrites out)
//   MatMulNTF64 out (+)= a @ btᵀ            (bt ALREADY [N,K] row-major — zero transpose; dA half)
//   MatMulTNF64 db  (+)= aᵀ @ dout          (dedicated column kernel — zero transpose; dB half)
// The NT/TN pair exists because the TS backward used to transpose B and A on the JS side (single-
// threaded) just so it could call the forward entry — pure overhead, since dOut @ Bᵀ wants exactly
// the [N,K]-style layout B is already stored in, and Aᵀ @ dOut can read A's columns in place.
//
// NUMERICS: no longer bit-identical to the TS path (see matmul_avx.c) — vectorizing the k loop keeps
// 4 partial sums and FMA rounds once instead of twice. This was an explicit trade for the SIMD win;
// gradcheck's 1e-3 tolerance is the correctness guard.

package main

/*
#cgo CFLAGS: -O3
// Dispatchers (matmul_dispatch.c): AVX2+FMA when the CPU has it, the scalar copy otherwise.
// NEVER declare/call the *Avx kernels here directly — they carry no CPUID check (SIGILL risk).
void MatMulRowsKernel(const double *A, const double *BT, double *O, int RowStart, int RowEnd, int K, int N, int Acc);
void MatMulTnKernel(const double *A, const double *DOut, double *DB, int PStart, int PEnd, int M, int K, int N, int Acc);
void MatMulRowsKernelF32(const float *A, const float *BT, float *O, int RowStart, int RowEnd, int K, int N, int Acc);
void MatMulTnKernelF32(const float *A, const float *DOut, float *DB, int PStart, int PEnd, int M, int K, int N, int Acc);
int CpuHasAvx2Fma(void);
void ForceScalarKernels(int Force);
*/
import "C"
import (
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"
)

// Rows of A the C kernel accumulates simultaneously against one BT row. Must match Block in
// matmul_avx.c — it is what makes the range split worth aligning.
const RowBlock = 8

// DB columns the TN kernel processes simultaneously (must match PBlock in matmul_avx.c). TN range
// splits snap to this so goroutine boundaries do not push columns into the slower remainder loop.
const TnColBlock = 8

// Transpose elements per goroutine (~32 KB of float64, about an L1 working set). Sets how wide the
// B-transpose fans out; see the worker-count comment in MatMulF64 for why it is size-driven.
const TransposeChunk = 4096

// FLOPs per goroutine for the TN fan-out (~50µs of FMA work per core). Same principle as
// TransposeChunk: worker count scales with WORK, not cores — measured on the transpose, fanning
// narrow shapes across all cores loses ~30% to spawn/join, so small backward shapes stay narrow.
const TnFlopChunk = 262144

// Reusable B-transpose scratch. The kernel is called thousands of times per training step and a fresh
// make() per call was pure allocator/GC churn (hundreds of KB each, at this model's shapes).
var BtPool = sync.Pool{
	New: func() any {
		Buf := make([]float64, 0)
		return &Buf
	},
}

// Same scratch pool for the f32 entry points (a shared pool would mix element types).
var BtPoolF32 = sync.Pool{
	New: func() any {
		Buf := make([]float32, 0)
		return &Buf
	},
}

// Process-global cap on every fan-out in this library (0 = runtime.NumCPU()). The training worker
// pool sets this to 1 while its JS workers run one kernel call each on their own thread — the pool
// IS the parallelism there, and per-call goroutine fan-out would only oversubscribe the cores. One
// Go runtime serves every JS worker thread in the process, so this is a process-wide phase switch,
// flipped by the pool around its dispatch (never mid-call).
var MaxKernelWorkers int64

// SetKernelThreads caps goroutine fan-out per kernel call; n <= 0 restores the all-cores default.
//
//export SetKernelThreads
func SetKernelThreads(n C.int) {
	atomic.StoreInt64(&MaxKernelWorkers, int64(n))
}

// ShahdCpuSupportsAvx2Fma reports whether the fast AVX2+FMA kernels run on this CPU (1) or the
// portable scalar fallback does (0). The TS BackendSelector calls it right after dlopen and prefers
// the TS backend when it is 0; a library WITHOUT this symbol predates the dispatcher and may SIGILL,
// so the selector treats its absence as unsafe.
//
//export ShahdCpuSupportsAvx2Fma
func ShahdCpuSupportsAvx2Fma() C.int {
	return C.CpuHasAvx2Fma()
}

// SetScalarKernels forces the scalar kernels (n != 0) or restores CPU-based dispatch (n == 0).
// Test hook: lets the scalar fallback be parity-checked on an AVX2 machine. Never enables AVX on a
// CPU without it.
//
//export SetScalarKernels
func SetScalarKernels(n C.int) {
	C.ForceScalarKernels(n)
}

func KernelWorkerCap() int {
	V := atomic.LoadInt64(&MaxKernelWorkers)
	if V <= 0 {
		return runtime.NumCPU()
	}
	return int(V)
}

// FanOutRows splits [0,M) into RowBlock-aligned ranges across goroutines and runs the C row kernel
// on each — one worker per CPU, capped by row count, each range snapped UP to a RowBlock multiple so
// workers run the C kernel's blocked fast path instead of straddling a block boundary (which would
// drop rows into its slower one-at-a-time remainder loop). bt must be [N,K] row-major.
func FanOutRows(a, bt, out *C.double, M, K, N, Acc int) {
	Workers := KernelWorkerCap()
	if Workers > M {
		Workers = M
	}
	if Workers < 1 {
		Workers = 1
	}
	RowsPer := (M + Workers - 1) / Workers
	RowsPer = ((RowsPer + RowBlock - 1) / RowBlock) * RowBlock

	// Single range: run on the calling thread. Under the worker pool this is EVERY call (cap = 1),
	// so the spawn/join would otherwise be paid thousands of times per step for zero parallelism.
	if RowsPer >= M {
		C.MatMulRowsKernel(a, bt, out, C.int(0), C.int(M), C.int(K), C.int(N), C.int(Acc))
		return
	}

	var Wg sync.WaitGroup
	for Start := 0; Start < M; Start += RowsPer {
		End := Start + RowsPer
		if End > M {
			End = M
		}
		Wg.Add(1)
		go func(RowStart, RowEnd int) {
			defer Wg.Done()
			C.MatMulRowsKernel(a, bt, out, C.int(RowStart), C.int(RowEnd), C.int(K), C.int(N), C.int(Acc))
		}(Start, End)
	}
	Wg.Wait()
}

// TransposeBt fills BT[N,K] from B[K,N] IN PARALLEL: a transpose is pure memory traffic (65k+
// elements at this model's shapes) and running it on one thread before the workers start made it
// Amdahl's serial fraction INSIDE the kernel. Splitting by output row (j) gives each worker a
// contiguous BT write range, so cores do not fight over the same cache lines.
//
// Worker count scales with transpose SIZE, not CPU count. Measured: fanning the wide MLP/LM-head
// transposes (~65k elements) across all cores wins ~1.4-1.6x, but doing the same to the narrow
// attention projections (~4k elements) LOSES ~30% — goroutine spawn/join costs more than the
// transpose itself at that size. One worker per TransposeChunk elements lets each shape pick.
func TransposeBt[T float64 | float32](BT, B []T, K, N int) {
	Workers := KernelWorkerCap()
	TWorkers := (N * K) / TransposeChunk
	if TWorkers > Workers {
		TWorkers = Workers
	}
	if TWorkers > N {
		TWorkers = N
	}
	if TWorkers < 1 {
		TWorkers = 1
	}
	if TWorkers == 1 { // same single-range rule as FanOutRows: no goroutine for no parallelism
		for j := 0; j < N; j++ {
			jK := j * K
			for p := 0; p < K; p++ {
				BT[jK+p] = B[p*N+j]
			}
		}
		return
	}
	JPer := (N + TWorkers - 1) / TWorkers
	var TWg sync.WaitGroup
	for J0 := 0; J0 < N; J0 += JPer {
		J1 := J0 + JPer
		if J1 > N {
			J1 = N
		}
		TWg.Add(1)
		go func(JStart, JEnd int) {
			defer TWg.Done()
			for j := JStart; j < JEnd; j++ {
				jK := j * K
				for p := 0; p < K; p++ {
					BT[jK+p] = B[p*N+j]
				}
			}
		}(J0, J1)
	}
	TWg.Wait()
}

// MatMulF64 computes out[M,N] = a[M,K] @ b[K,N], row-major. out is fully overwritten.
//
//export MatMulF64
func MatMulF64(a *C.double, b *C.double, out *C.double, m C.int, k C.int, n C.int) {
	M := int(m)
	K := int(k)
	N := int(n)
	B := unsafe.Slice((*float64)(unsafe.Pointer(b)), K*N)

	BtRef := BtPool.Get().(*[]float64)
	if cap(*BtRef) < N*K {
		*BtRef = make([]float64, N*K)
	}
	BT := (*BtRef)[:N*K]
	TransposeBt(BT, B, K, N)
	BtPtr := (*C.double)(unsafe.Pointer(&BT[0])) // Go memory, but C never retains it past the call

	FanOutRows(a, BtPtr, out, M, K, N, 0)
	BtPool.Put(BtRef) // safe: every worker has joined, so nothing still reads BT
}

// MatMulNTF64 computes out[M,N] (+)= a[M,K] @ btᵀ where bt is ALREADY stored [N,K] row-major —
// exactly the layout the row kernel streams, so NO transpose happens at all. This is the dA half
// of the matmul backward: dA[M,K] += dOut[M,N] @ Bᵀ, where the stored B[K,N] IS the [K,N]-shaped
// pre-transposed operand. acc != 0 accumulates into out instead of overwriting.
//
//export MatMulNTF64
func MatMulNTF64(a *C.double, bt *C.double, out *C.double, m C.int, k C.int, n C.int, acc C.int) {
	FanOutRows(a, bt, out, int(m), int(k), int(n), int(acc))
}

// MatMulTNF64 computes db[K,N] (+)= aᵀ @ dout where a is [M,K] and dout is [M,N] — the dB half of
// the matmul backward (dB += Aᵀ @ dOut), again with zero transposes: the C kernel reads a's column
// p in place while DB[p,:] stays hot in L1. Fan-out is over db ROWS (p ranges), so every goroutine
// writes a disjoint db range and only shares reads.
//
//export MatMulTNF64
func MatMulTNF64(a *C.double, dout *C.double, db *C.double, m C.int, k C.int, n C.int, acc C.int) {
	M := int(m)
	K := int(k)
	N := int(n)

	Workers := (M * K * N) / TnFlopChunk
	Cpus := KernelWorkerCap()
	if Workers > Cpus {
		Workers = Cpus
	}
	if Workers > K {
		Workers = K
	}
	if Workers < 1 {
		Workers = 1
	}
	if Workers == 1 { // same single-range rule as FanOutRows
		C.MatMulTnKernel(a, dout, db, C.int(0), C.int(K), C.int(M), C.int(K), C.int(N), C.int(acc))
		return
	}
	PPer := (K + Workers - 1) / Workers
	PPer = ((PPer + TnColBlock - 1) / TnColBlock) * TnColBlock

	var Wg sync.WaitGroup
	for P0 := 0; P0 < K; P0 += PPer {
		P1 := P0 + PPer
		if P1 > K {
			P1 = K
		}
		Wg.Add(1)
		go func(PStart, PEnd int) {
			defer Wg.Done()
			C.MatMulTnKernel(a, dout, db, C.int(PStart), C.int(PEnd), C.int(M), C.int(K), C.int(N), C.int(acc))
		}(P0, P1)
	}
	Wg.Wait()
}

// ── F32 entry points — the same threading shell over the f32 C kernels (see matmul_avx.c: AVX2 is
// 8 lanes of f32 vs 4 of f64, and the narrow shapes are memory-bound, so f32 doubles FLOPs per
// instruction AND halves bytes moved). Fan-out mirrors the f64 trio; TransposeBt is generic.

// FanOutRowsF32 mirrors FanOutRows for the f32 row kernel (same RowBlock — both kernels block 8 rows).
func FanOutRowsF32(a, bt, out *C.float, M, K, N, Acc int) {
	Workers := KernelWorkerCap()
	if Workers > M {
		Workers = M
	}
	if Workers < 1 {
		Workers = 1
	}
	RowsPer := (M + Workers - 1) / Workers
	RowsPer = ((RowsPer + RowBlock - 1) / RowBlock) * RowBlock

	if RowsPer >= M { // single range: run on the calling thread (the worker-pool fast path)
		C.MatMulRowsKernelF32(a, bt, out, C.int(0), C.int(M), C.int(K), C.int(N), C.int(Acc))
		return
	}

	var Wg sync.WaitGroup
	for Start := 0; Start < M; Start += RowsPer {
		End := Start + RowsPer
		if End > M {
			End = M
		}
		Wg.Add(1)
		go func(RowStart, RowEnd int) {
			defer Wg.Done()
			C.MatMulRowsKernelF32(a, bt, out, C.int(RowStart), C.int(RowEnd), C.int(K), C.int(N), C.int(Acc))
		}(Start, End)
	}
	Wg.Wait()
}

// MatMulF32 computes out[M,N] = a[M,K] @ b[K,N], row-major, in f32. out is fully overwritten.
//
//export MatMulF32
func MatMulF32(a *C.float, b *C.float, out *C.float, m C.int, k C.int, n C.int) {
	M := int(m)
	K := int(k)
	N := int(n)
	B := unsafe.Slice((*float32)(unsafe.Pointer(b)), K*N)

	BtRef := BtPoolF32.Get().(*[]float32)
	if cap(*BtRef) < N*K {
		*BtRef = make([]float32, N*K)
	}
	BT := (*BtRef)[:N*K]
	TransposeBt(BT, B, K, N)
	BtPtr := (*C.float)(unsafe.Pointer(&BT[0])) // Go memory, but C never retains it past the call

	FanOutRowsF32(a, BtPtr, out, M, K, N, 0)
	BtPoolF32.Put(BtRef) // safe: every worker has joined, so nothing still reads BT
}

// MatMulNTF32 computes out[M,N] (+)= a[M,K] @ btᵀ with bt ALREADY [N,K] row-major (dA half, f32).
//
//export MatMulNTF32
func MatMulNTF32(a *C.float, bt *C.float, out *C.float, m C.int, k C.int, n C.int, acc C.int) {
	FanOutRowsF32(a, bt, out, int(m), int(k), int(n), int(acc))
}

// MatMulTNF32 computes db[K,N] (+)= aᵀ @ dout (dB half, f32) — fan-out over db row ranges, same
// work-scaled worker count as the f64 version.
//
//export MatMulTNF32
func MatMulTNF32(a *C.float, dout *C.float, db *C.float, m C.int, k C.int, n C.int, acc C.int) {
	M := int(m)
	K := int(k)
	N := int(n)

	Workers := (M * K * N) / TnFlopChunk
	Cpus := KernelWorkerCap()
	if Workers > Cpus {
		Workers = Cpus
	}
	if Workers > K {
		Workers = K
	}
	if Workers < 1 {
		Workers = 1
	}
	if Workers == 1 { // same single-range rule as FanOutRows
		C.MatMulTnKernelF32(a, dout, db, C.int(0), C.int(K), C.int(M), C.int(K), C.int(N), C.int(acc))
		return
	}
	PPer := (K + Workers - 1) / Workers
	PPer = ((PPer + TnColBlock - 1) / TnColBlock) * TnColBlock

	var Wg sync.WaitGroup
	for P0 := 0; P0 < K; P0 += PPer {
		P1 := P0 + PPer
		if P1 > K {
			P1 = K
		}
		Wg.Add(1)
		go func(PStart, PEnd int) {
			defer Wg.Done()
			C.MatMulTnKernelF32(a, dout, db, C.int(PStart), C.int(PEnd), C.int(M), C.int(K), C.int(N), C.int(acc))
		}(P0, P1)
	}
	Wg.Wait()
}

func main() {}
