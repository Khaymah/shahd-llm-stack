// AVX2 + FMA micro-kernel for the row range of a matmul. Lives in its own .c file because cgo forbids
// DEFINITIONS in the preamble of a file that uses //export (matmul.go only declares this).
//
// Why C and not Go: Go has no SIMD intrinsics — a Go kernel is scalar unless hand-written in Plan9
// assembly, which is why the previous scalar Go kernel ran at ~16 GFLOP/s on a CPU capable of ~500.
// gcc with -mavx2 -mfma gives 4 doubles per instruction plus fused multiply-add, verified present on
// this machine (Broadwell: -mavx2/-mfma both [enabled] under -march=native).
//
// Three levers stack here:
//   1. B is pre-transposed by the caller (BT[N,K]) so A[i,:] and BT[j,:] both stream contiguously.
//   2. Register blocking over 8 rows of A: each BT[j,:] vector is loaded ONCE and reused by all 8
//      accumulators, cutting BT memory traffic ~8x. (This kernel is memory-bandwidth bound without it:
//      the naive form re-reads all of BT for every row of A.)
//   3. 4-wide FMA down the k loop.
//
// NUMERICS: this is deliberately NOT bit-identical to the scalar TS path. Vectorizing the k loop keeps
// 4 partial sums that are combined at the end, and FMA rounds once instead of twice. Both change the
// last bits. That trade was made explicitly to buy the SIMD speedup; gradcheck's 1e-3 finite-difference
// tolerance remains the correctness guard.

// RUNTIME SAFETY: nothing here checks CPUID — these functions are only ever reached through the
// dispatchers in matmul_dispatch.c, which route to the scalar copies in matmul_scalar.c when the
// CPU lacks AVX2/FMA. Never call these directly from matmul.go (that would bring back the SIGILL).
// The whole file compiles to nothing on non-x86 targets, where the scalar kernels are the only path.

#if defined(__x86_64__) || defined(__i386__)

#include <immintrin.h>
#include <stddef.h>

// The ISA target is pinned HERE rather than via #cgo CFLAGS because cgo rejects -mfma as an unsafe
// flag (go.dev/s/invalidflag). Pinning it in the source also keeps the build a plain
// `go build -buildmode=c-shared` with no environment variables to remember.
#pragma GCC push_options
#pragma GCC target("avx2,fma")

// Horizontal sum of a 4-lane vector. Fixed lane order (0+2, 1+3, then the pair) so the result is
// deterministic run to run.
static inline double HSum256(__m256d V) {
    __m128d Lo = _mm256_castpd256_pd128(V);
    __m128d Hi = _mm256_extractf128_pd(V, 1);
    Lo = _mm_add_pd(Lo, Hi);
    __m128d Sh = _mm_unpackhi_pd(Lo, Lo);
    return _mm_cvtsd_f64(_mm_add_sd(Lo, Sh));
}

// O[i,j] = dot(A[i,:], BT[j,:]) for i in [RowStart, RowEnd), all j in [0, N).
// Acc != 0 accumulates (O[i,j] += ...) instead of overwriting — the backward pass adds gradient
// contributions into buffers that already hold earlier sequences' gradients (grad accumulation).
void MatMulRowsAvx(const double *A, const double *BT, double *O,
                   int RowStart, int RowEnd, int K, int N, int Acc) {
    const int Block = 8;
    int I = RowStart;

    for (; I + Block <= RowEnd; I += Block) {
        const double *A0 = A + (size_t)(I + 0) * K;
        const double *A1 = A + (size_t)(I + 1) * K;
        const double *A2 = A + (size_t)(I + 2) * K;
        const double *A3 = A + (size_t)(I + 3) * K;
        const double *A4 = A + (size_t)(I + 4) * K;
        const double *A5 = A + (size_t)(I + 5) * K;
        const double *A6 = A + (size_t)(I + 6) * K;
        const double *A7 = A + (size_t)(I + 7) * K;

        for (int J = 0; J < N; J++) {
            const double *Bt = BT + (size_t)J * K;
            __m256d S0 = _mm256_setzero_pd(), S1 = _mm256_setzero_pd();
            __m256d S2 = _mm256_setzero_pd(), S3 = _mm256_setzero_pd();
            __m256d S4 = _mm256_setzero_pd(), S5 = _mm256_setzero_pd();
            __m256d S6 = _mm256_setzero_pd(), S7 = _mm256_setzero_pd();

            int P = 0;
            for (; P + 4 <= K; P += 4) {
                __m256d Bv = _mm256_loadu_pd(Bt + P); // loaded once, reused by all 8 rows
                S0 = _mm256_fmadd_pd(_mm256_loadu_pd(A0 + P), Bv, S0);
                S1 = _mm256_fmadd_pd(_mm256_loadu_pd(A1 + P), Bv, S1);
                S2 = _mm256_fmadd_pd(_mm256_loadu_pd(A2 + P), Bv, S2);
                S3 = _mm256_fmadd_pd(_mm256_loadu_pd(A3 + P), Bv, S3);
                S4 = _mm256_fmadd_pd(_mm256_loadu_pd(A4 + P), Bv, S4);
                S5 = _mm256_fmadd_pd(_mm256_loadu_pd(A5 + P), Bv, S5);
                S6 = _mm256_fmadd_pd(_mm256_loadu_pd(A6 + P), Bv, S6);
                S7 = _mm256_fmadd_pd(_mm256_loadu_pd(A7 + P), Bv, S7);
            }

            double R0 = HSum256(S0), R1 = HSum256(S1), R2 = HSum256(S2), R3 = HSum256(S3);
            double R4 = HSum256(S4), R5 = HSum256(S5), R6 = HSum256(S6), R7 = HSum256(S7);
            for (; P < K; P++) { // K % 4 tail
                double Bv = Bt[P];
                R0 += A0[P] * Bv; R1 += A1[P] * Bv; R2 += A2[P] * Bv; R3 += A3[P] * Bv;
                R4 += A4[P] * Bv; R5 += A5[P] * Bv; R6 += A6[P] * Bv; R7 += A7[P] * Bv;
            }

            if (Acc) {
                O[(size_t)(I + 0) * N + J] += R0;
                O[(size_t)(I + 1) * N + J] += R1;
                O[(size_t)(I + 2) * N + J] += R2;
                O[(size_t)(I + 3) * N + J] += R3;
                O[(size_t)(I + 4) * N + J] += R4;
                O[(size_t)(I + 5) * N + J] += R5;
                O[(size_t)(I + 6) * N + J] += R6;
                O[(size_t)(I + 7) * N + J] += R7;
            } else {
                O[(size_t)(I + 0) * N + J] = R0;
                O[(size_t)(I + 1) * N + J] = R1;
                O[(size_t)(I + 2) * N + J] = R2;
                O[(size_t)(I + 3) * N + J] = R3;
                O[(size_t)(I + 4) * N + J] = R4;
                O[(size_t)(I + 5) * N + J] = R5;
                O[(size_t)(I + 6) * N + J] = R6;
                O[(size_t)(I + 7) * N + J] = R7;
            }
        }
    }

    // Rows left over when the range is not a multiple of Block: same vectorized dot, one row at a time.
    for (; I < RowEnd; I++) {
        const double *Ai = A + (size_t)I * K;
        for (int J = 0; J < N; J++) {
            const double *Bt = BT + (size_t)J * K;
            __m256d S = _mm256_setzero_pd();
            int P = 0;
            for (; P + 4 <= K; P += 4) {
                S = _mm256_fmadd_pd(_mm256_loadu_pd(Ai + P), _mm256_loadu_pd(Bt + P), S);
            }
            double R = HSum256(S);
            for (; P < K; P++) R += Ai[P] * Bt[P];
            if (Acc) O[(size_t)I * N + J] += R;
            else O[(size_t)I * N + J] = R;
        }
    }
}

// DB[p,j] (+)= sum_i A[i,p] * DOut[i,j] for p in [PStart, PEnd), all j — i.e. DB = Aᵀ @ DOut
// without EITHER operand being transposed in memory. This is the dB half of the matmul backward
// (dB += Aᵀ @ dOut); the dA half reuses MatMulRowsAvx, because dOut @ Bᵀ wants exactly the BT
// layout B is already stored in.
//
// Loop order p → i → j: DB[p,:] stays hot in L1 across the whole i loop (N ≤ a few thousand
// doubles at this model's shapes) while DOut streams row by row. Broadcast-FMA down j means no
// horizontal sums at all. Per (p,j) the sum runs over i ASCENDING — the same order as the scalar
// TS backward — so the only numeric drift vs TS is FMA's single rounding.
//
// BLOCKED over PBlock=8 columns of A: one DOut[i,:] load feeds 8 DB rows, cutting DOut re-reads
// 8x. Profiled BEFORE this blocking, the p-at-a-time form re-streamed the whole DOut matrix once
// per p (~1GB per Nano sequence) and was 57% of all matmul time under the single-threaded worker
// regime. The 8 DB rows (<=32KB at this model's N) stay cache-hot across the i loop. Per (p,j)
// the sum still runs over i ASCENDING, so the blocked form is BIT-IDENTICAL to the row-at-a-time
// form for finite inputs (0.0 * x contributes exactly nothing) — only the remainder path keeps
// the Av==0 skip, where it is still worth one compare.
void MatMulTnAvx(const double *A, const double *DOut, double *DB,
                 int PStart, int PEnd, int M, int K, int N, int Acc) {
    // Block width adapts to N: the blocked pass read-modify-writes PBlock DB rows per i, and that
    // working set must SHARE L1 with the streaming DOut row. 8 rows x N=512 doubles = 32KB filled
    // L1 exactly and measured SLOWER than no blocking at all; 4 rows (16KB) leave room. Narrow
    // shapes (N <= 256, the attention projections/scores) keep the full 8-wide reuse.
    int P = PStart;

    if (N > 256) {
        for (; P + 4 <= PEnd; P += 4) {
            double *Db0 = DB + (size_t)(P + 0) * N;
            double *Db1 = DB + (size_t)(P + 1) * N;
            double *Db2 = DB + (size_t)(P + 2) * N;
            double *Db3 = DB + (size_t)(P + 3) * N;
            if (!Acc) {
                for (int J = 0; J < N; J++) { Db0[J] = 0.0; Db1[J] = 0.0; Db2[J] = 0.0; Db3[J] = 0.0; }
            }
            for (int I = 0; I < M; I++) {
                const double *Ar = A + (size_t)I * K + P;
                const double *Dr = DOut + (size_t)I * N;
                __m256d V0 = _mm256_set1_pd(Ar[0]), V1 = _mm256_set1_pd(Ar[1]);
                __m256d V2 = _mm256_set1_pd(Ar[2]), V3 = _mm256_set1_pd(Ar[3]);
                int J = 0;
                for (; J + 4 <= N; J += 4) {
                    __m256d Dv = _mm256_loadu_pd(Dr + J); // loaded once, reused by 4 DB rows
                    _mm256_storeu_pd(Db0 + J, _mm256_fmadd_pd(V0, Dv, _mm256_loadu_pd(Db0 + J)));
                    _mm256_storeu_pd(Db1 + J, _mm256_fmadd_pd(V1, Dv, _mm256_loadu_pd(Db1 + J)));
                    _mm256_storeu_pd(Db2 + J, _mm256_fmadd_pd(V2, Dv, _mm256_loadu_pd(Db2 + J)));
                    _mm256_storeu_pd(Db3 + J, _mm256_fmadd_pd(V3, Dv, _mm256_loadu_pd(Db3 + J)));
                }
                for (; J < N; J++) { // N % 4 tail
                    double Dj = Dr[J];
                    Db0[J] += Ar[0] * Dj; Db1[J] += Ar[1] * Dj; Db2[J] += Ar[2] * Dj; Db3[J] += Ar[3] * Dj;
                }
            }
        }
    }

    for (; P + 8 <= PEnd; P += 8) {
        double *Db0 = DB + (size_t)(P + 0) * N;
        double *Db1 = DB + (size_t)(P + 1) * N;
        double *Db2 = DB + (size_t)(P + 2) * N;
        double *Db3 = DB + (size_t)(P + 3) * N;
        double *Db4 = DB + (size_t)(P + 4) * N;
        double *Db5 = DB + (size_t)(P + 5) * N;
        double *Db6 = DB + (size_t)(P + 6) * N;
        double *Db7 = DB + (size_t)(P + 7) * N;
        if (!Acc) {
            for (int J = 0; J < N; J++) {
                Db0[J] = 0.0; Db1[J] = 0.0; Db2[J] = 0.0; Db3[J] = 0.0;
                Db4[J] = 0.0; Db5[J] = 0.0; Db6[J] = 0.0; Db7[J] = 0.0;
            }
        }
        for (int I = 0; I < M; I++) {
            const double *Ar = A + (size_t)I * K + P;
            const double *Dr = DOut + (size_t)I * N;
            __m256d V0 = _mm256_set1_pd(Ar[0]), V1 = _mm256_set1_pd(Ar[1]);
            __m256d V2 = _mm256_set1_pd(Ar[2]), V3 = _mm256_set1_pd(Ar[3]);
            __m256d V4 = _mm256_set1_pd(Ar[4]), V5 = _mm256_set1_pd(Ar[5]);
            __m256d V6 = _mm256_set1_pd(Ar[6]), V7 = _mm256_set1_pd(Ar[7]);
            int J = 0;
            for (; J + 4 <= N; J += 4) {
                __m256d Dv = _mm256_loadu_pd(Dr + J); // loaded once, reused by all 8 DB rows
                _mm256_storeu_pd(Db0 + J, _mm256_fmadd_pd(V0, Dv, _mm256_loadu_pd(Db0 + J)));
                _mm256_storeu_pd(Db1 + J, _mm256_fmadd_pd(V1, Dv, _mm256_loadu_pd(Db1 + J)));
                _mm256_storeu_pd(Db2 + J, _mm256_fmadd_pd(V2, Dv, _mm256_loadu_pd(Db2 + J)));
                _mm256_storeu_pd(Db3 + J, _mm256_fmadd_pd(V3, Dv, _mm256_loadu_pd(Db3 + J)));
                _mm256_storeu_pd(Db4 + J, _mm256_fmadd_pd(V4, Dv, _mm256_loadu_pd(Db4 + J)));
                _mm256_storeu_pd(Db5 + J, _mm256_fmadd_pd(V5, Dv, _mm256_loadu_pd(Db5 + J)));
                _mm256_storeu_pd(Db6 + J, _mm256_fmadd_pd(V6, Dv, _mm256_loadu_pd(Db6 + J)));
                _mm256_storeu_pd(Db7 + J, _mm256_fmadd_pd(V7, Dv, _mm256_loadu_pd(Db7 + J)));
            }
            for (; J < N; J++) { // N % 4 tail
                double Dj = Dr[J];
                Db0[J] += Ar[0] * Dj; Db1[J] += Ar[1] * Dj; Db2[J] += Ar[2] * Dj; Db3[J] += Ar[3] * Dj;
                Db4[J] += Ar[4] * Dj; Db5[J] += Ar[5] * Dj; Db6[J] += Ar[6] * Dj; Db7[J] += Ar[7] * Dj;
            }
        }
    }

    // Columns left over when the range is not a multiple of PBlock: the original one-at-a-time form.
    for (; P < PEnd; P++) {
        double *Db = DB + (size_t)P * N;
        if (!Acc) {
            for (int J = 0; J < N; J++) Db[J] = 0.0;
        }
        for (int I = 0; I < M; I++) {
            double Av = A[(size_t)I * K + P];
            if (Av == 0.0) continue;
            const double *Dr = DOut + (size_t)I * N;
            __m256d Vv = _mm256_set1_pd(Av);
            int J = 0;
            for (; J + 4 <= N; J += 4) {
                __m256d D = _mm256_loadu_pd(Db + J);
                D = _mm256_fmadd_pd(Vv, _mm256_loadu_pd(Dr + J), D);
                _mm256_storeu_pd(Db + J, D);
            }
            for (; J < N; J++) Db[J] += Av * Dr[J];
        }
    }
}

// ── F32 kernels: the same three levers at DOUBLE the SIMD width ──────────────────────────────────
// AVX2 is 4 lanes of f64 but 8 lanes of f32, and the narrow shapes are memory-bound — so f32 both
// doubles FLOPs per instruction AND halves bytes moved. Structure mirrors the f64 kernels exactly
// (8-row register blocking / broadcast-FMA TN with adaptive column blocking); only the lane width,
// the tail strides, and the TN blocking threshold change (8 DB rows x 512 floats = 16KB, so the
// full 8-wide reuse stays safe up to N=512; the 4-wide fallback starts above that).

// Horizontal sum of an 8-lane float vector. Fixed reduction order (halves, then pairs, then the
// final pair) so the result is deterministic run to run.
static inline float HSum256F(__m256 V) {
    __m128 Lo = _mm256_castps256_ps128(V);
    __m128 Hi = _mm256_extractf128_ps(V, 1);
    Lo = _mm_add_ps(Lo, Hi);
    __m128 Sh = _mm_movehl_ps(Lo, Lo);
    Lo = _mm_add_ps(Lo, Sh);
    Sh = _mm_shuffle_ps(Lo, Lo, 0x55);
    return _mm_cvtss_f32(_mm_add_ss(Lo, Sh));
}

// O[i,j] = dot(A[i,:], BT[j,:]) for i in [RowStart, RowEnd), all j in [0, N) — f32, 8 floats/FMA.
void MatMulRowsAvxF32(const float *A, const float *BT, float *O,
                      int RowStart, int RowEnd, int K, int N, int Acc) {
    const int Block = 8;
    int I = RowStart;

    for (; I + Block <= RowEnd; I += Block) {
        const float *A0 = A + (size_t)(I + 0) * K;
        const float *A1 = A + (size_t)(I + 1) * K;
        const float *A2 = A + (size_t)(I + 2) * K;
        const float *A3 = A + (size_t)(I + 3) * K;
        const float *A4 = A + (size_t)(I + 4) * K;
        const float *A5 = A + (size_t)(I + 5) * K;
        const float *A6 = A + (size_t)(I + 6) * K;
        const float *A7 = A + (size_t)(I + 7) * K;

        for (int J = 0; J < N; J++) {
            const float *Bt = BT + (size_t)J * K;
            __m256 S0 = _mm256_setzero_ps(), S1 = _mm256_setzero_ps();
            __m256 S2 = _mm256_setzero_ps(), S3 = _mm256_setzero_ps();
            __m256 S4 = _mm256_setzero_ps(), S5 = _mm256_setzero_ps();
            __m256 S6 = _mm256_setzero_ps(), S7 = _mm256_setzero_ps();

            int P = 0;
            for (; P + 8 <= K; P += 8) {
                __m256 Bv = _mm256_loadu_ps(Bt + P); // loaded once, reused by all 8 rows
                S0 = _mm256_fmadd_ps(_mm256_loadu_ps(A0 + P), Bv, S0);
                S1 = _mm256_fmadd_ps(_mm256_loadu_ps(A1 + P), Bv, S1);
                S2 = _mm256_fmadd_ps(_mm256_loadu_ps(A2 + P), Bv, S2);
                S3 = _mm256_fmadd_ps(_mm256_loadu_ps(A3 + P), Bv, S3);
                S4 = _mm256_fmadd_ps(_mm256_loadu_ps(A4 + P), Bv, S4);
                S5 = _mm256_fmadd_ps(_mm256_loadu_ps(A5 + P), Bv, S5);
                S6 = _mm256_fmadd_ps(_mm256_loadu_ps(A6 + P), Bv, S6);
                S7 = _mm256_fmadd_ps(_mm256_loadu_ps(A7 + P), Bv, S7);
            }

            float R0 = HSum256F(S0), R1 = HSum256F(S1), R2 = HSum256F(S2), R3 = HSum256F(S3);
            float R4 = HSum256F(S4), R5 = HSum256F(S5), R6 = HSum256F(S6), R7 = HSum256F(S7);
            for (; P < K; P++) { // K % 8 tail
                float Bv = Bt[P];
                R0 += A0[P] * Bv; R1 += A1[P] * Bv; R2 += A2[P] * Bv; R3 += A3[P] * Bv;
                R4 += A4[P] * Bv; R5 += A5[P] * Bv; R6 += A6[P] * Bv; R7 += A7[P] * Bv;
            }

            if (Acc) {
                O[(size_t)(I + 0) * N + J] += R0;
                O[(size_t)(I + 1) * N + J] += R1;
                O[(size_t)(I + 2) * N + J] += R2;
                O[(size_t)(I + 3) * N + J] += R3;
                O[(size_t)(I + 4) * N + J] += R4;
                O[(size_t)(I + 5) * N + J] += R5;
                O[(size_t)(I + 6) * N + J] += R6;
                O[(size_t)(I + 7) * N + J] += R7;
            } else {
                O[(size_t)(I + 0) * N + J] = R0;
                O[(size_t)(I + 1) * N + J] = R1;
                O[(size_t)(I + 2) * N + J] = R2;
                O[(size_t)(I + 3) * N + J] = R3;
                O[(size_t)(I + 4) * N + J] = R4;
                O[(size_t)(I + 5) * N + J] = R5;
                O[(size_t)(I + 6) * N + J] = R6;
                O[(size_t)(I + 7) * N + J] = R7;
            }
        }
    }

    for (; I < RowEnd; I++) { // rows left over when the range is not a multiple of Block
        const float *Ai = A + (size_t)I * K;
        for (int J = 0; J < N; J++) {
            const float *Bt = BT + (size_t)J * K;
            __m256 S = _mm256_setzero_ps();
            int P = 0;
            for (; P + 8 <= K; P += 8) {
                S = _mm256_fmadd_ps(_mm256_loadu_ps(Ai + P), _mm256_loadu_ps(Bt + P), S);
            }
            float R = HSum256F(S);
            for (; P < K; P++) R += Ai[P] * Bt[P];
            if (Acc) O[(size_t)I * N + J] += R;
            else O[(size_t)I * N + J] = R;
        }
    }
}

// DB[p,j] (+)= sum_i A[i,p] * DOut[i,j] — the f32 TN kernel (dB half of the backward). Same
// adaptive column blocking as the f64 version, but the L1 threshold doubles: 8 DB rows of floats
// are half the bytes, so the 8-wide reuse holds through N=512 and the 4-wide path starts above.
void MatMulTnAvxF32(const float *A, const float *DOut, float *DB,
                    int PStart, int PEnd, int M, int K, int N, int Acc) {
    int P = PStart;

    if (N > 512) {
        for (; P + 4 <= PEnd; P += 4) {
            float *Db0 = DB + (size_t)(P + 0) * N;
            float *Db1 = DB + (size_t)(P + 1) * N;
            float *Db2 = DB + (size_t)(P + 2) * N;
            float *Db3 = DB + (size_t)(P + 3) * N;
            if (!Acc) {
                for (int J = 0; J < N; J++) { Db0[J] = 0.0f; Db1[J] = 0.0f; Db2[J] = 0.0f; Db3[J] = 0.0f; }
            }
            for (int I = 0; I < M; I++) {
                const float *Ar = A + (size_t)I * K + P;
                const float *Dr = DOut + (size_t)I * N;
                __m256 V0 = _mm256_set1_ps(Ar[0]), V1 = _mm256_set1_ps(Ar[1]);
                __m256 V2 = _mm256_set1_ps(Ar[2]), V3 = _mm256_set1_ps(Ar[3]);
                int J = 0;
                for (; J + 8 <= N; J += 8) {
                    __m256 Dv = _mm256_loadu_ps(Dr + J); // loaded once, reused by 4 DB rows
                    _mm256_storeu_ps(Db0 + J, _mm256_fmadd_ps(V0, Dv, _mm256_loadu_ps(Db0 + J)));
                    _mm256_storeu_ps(Db1 + J, _mm256_fmadd_ps(V1, Dv, _mm256_loadu_ps(Db1 + J)));
                    _mm256_storeu_ps(Db2 + J, _mm256_fmadd_ps(V2, Dv, _mm256_loadu_ps(Db2 + J)));
                    _mm256_storeu_ps(Db3 + J, _mm256_fmadd_ps(V3, Dv, _mm256_loadu_ps(Db3 + J)));
                }
                for (; J < N; J++) { // N % 8 tail
                    float Dj = Dr[J];
                    Db0[J] += Ar[0] * Dj; Db1[J] += Ar[1] * Dj; Db2[J] += Ar[2] * Dj; Db3[J] += Ar[3] * Dj;
                }
            }
        }
    }

    for (; P + 8 <= PEnd; P += 8) {
        float *Db0 = DB + (size_t)(P + 0) * N;
        float *Db1 = DB + (size_t)(P + 1) * N;
        float *Db2 = DB + (size_t)(P + 2) * N;
        float *Db3 = DB + (size_t)(P + 3) * N;
        float *Db4 = DB + (size_t)(P + 4) * N;
        float *Db5 = DB + (size_t)(P + 5) * N;
        float *Db6 = DB + (size_t)(P + 6) * N;
        float *Db7 = DB + (size_t)(P + 7) * N;
        if (!Acc) {
            for (int J = 0; J < N; J++) {
                Db0[J] = 0.0f; Db1[J] = 0.0f; Db2[J] = 0.0f; Db3[J] = 0.0f;
                Db4[J] = 0.0f; Db5[J] = 0.0f; Db6[J] = 0.0f; Db7[J] = 0.0f;
            }
        }
        for (int I = 0; I < M; I++) {
            const float *Ar = A + (size_t)I * K + P;
            const float *Dr = DOut + (size_t)I * N;
            __m256 V0 = _mm256_set1_ps(Ar[0]), V1 = _mm256_set1_ps(Ar[1]);
            __m256 V2 = _mm256_set1_ps(Ar[2]), V3 = _mm256_set1_ps(Ar[3]);
            __m256 V4 = _mm256_set1_ps(Ar[4]), V5 = _mm256_set1_ps(Ar[5]);
            __m256 V6 = _mm256_set1_ps(Ar[6]), V7 = _mm256_set1_ps(Ar[7]);
            int J = 0;
            for (; J + 8 <= N; J += 8) {
                __m256 Dv = _mm256_loadu_ps(Dr + J); // loaded once, reused by all 8 DB rows
                _mm256_storeu_ps(Db0 + J, _mm256_fmadd_ps(V0, Dv, _mm256_loadu_ps(Db0 + J)));
                _mm256_storeu_ps(Db1 + J, _mm256_fmadd_ps(V1, Dv, _mm256_loadu_ps(Db1 + J)));
                _mm256_storeu_ps(Db2 + J, _mm256_fmadd_ps(V2, Dv, _mm256_loadu_ps(Db2 + J)));
                _mm256_storeu_ps(Db3 + J, _mm256_fmadd_ps(V3, Dv, _mm256_loadu_ps(Db3 + J)));
                _mm256_storeu_ps(Db4 + J, _mm256_fmadd_ps(V4, Dv, _mm256_loadu_ps(Db4 + J)));
                _mm256_storeu_ps(Db5 + J, _mm256_fmadd_ps(V5, Dv, _mm256_loadu_ps(Db5 + J)));
                _mm256_storeu_ps(Db6 + J, _mm256_fmadd_ps(V6, Dv, _mm256_loadu_ps(Db6 + J)));
                _mm256_storeu_ps(Db7 + J, _mm256_fmadd_ps(V7, Dv, _mm256_loadu_ps(Db7 + J)));
            }
            for (; J < N; J++) { // N % 8 tail
                float Dj = Dr[J];
                Db0[J] += Ar[0] * Dj; Db1[J] += Ar[1] * Dj; Db2[J] += Ar[2] * Dj; Db3[J] += Ar[3] * Dj;
                Db4[J] += Ar[4] * Dj; Db5[J] += Ar[5] * Dj; Db6[J] += Ar[6] * Dj; Db7[J] += Ar[7] * Dj;
            }
        }
    }

    for (; P < PEnd; P++) { // columns left over when the range is not a multiple of the block
        float *Db = DB + (size_t)P * N;
        if (!Acc) {
            for (int J = 0; J < N; J++) Db[J] = 0.0f;
        }
        for (int I = 0; I < M; I++) {
            float Av = A[(size_t)I * K + P];
            if (Av == 0.0f) continue;
            const float *Dr = DOut + (size_t)I * N;
            __m256 Vv = _mm256_set1_ps(Av);
            int J = 0;
            for (; J + 8 <= N; J += 8) {
                __m256 D = _mm256_loadu_ps(Db + J);
                D = _mm256_fmadd_ps(Vv, _mm256_loadu_ps(Dr + J), D);
                _mm256_storeu_ps(Db + J, D);
            }
            for (; J < N; J++) Db[J] += Av * Dr[J];
        }
    }
}

#pragma GCC pop_options

#endif // x86
