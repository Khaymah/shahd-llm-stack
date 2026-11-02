// Portable scalar copies of the four matmul kernels — the fallback the dispatcher in
// matmul_dispatch.c picks when the CPU (or the OS) does not support AVX2+FMA, and the ONLY path on
// non-x86 targets. Compiled for the baseline ISA (no target pragma), so loading or calling anything
// here can never raise SIGILL.
//
// Semantics are identical to the AVX kernels in matmul_avx.c (same index ranges, same Acc meaning);
// only the arithmetic order differs (one running sum per output instead of 4/8 SIMD partials), so
// results agree with the AVX path to f64/f32 accumulation noise, not bit-for-bit. The guard is
// Tests/FfiGuard.Test.ts, which forces this path via SetScalarKernels and checks it against TS.

#include <stddef.h>

// O[i,j] (+)= dot(A[i,:], BT[j,:]) for i in [RowStart, RowEnd), all j in [0, N). BT is [N,K].
void MatMulRowsScalar(const double *A, const double *BT, double *O,
                      int RowStart, int RowEnd, int K, int N, int Acc) {
    for (int I = RowStart; I < RowEnd; I++) {
        const double *Ai = A + (size_t)I * K;
        double *Oi = O + (size_t)I * N;
        for (int J = 0; J < N; J++) {
            const double *Bt = BT + (size_t)J * K;
            double R = 0.0;
            for (int P = 0; P < K; P++) R += Ai[P] * Bt[P];
            if (Acc) Oi[J] += R;
            else Oi[J] = R;
        }
    }
}

// DB[p,j] (+)= sum_i A[i,p] * DOut[i,j] for p in [PStart, PEnd), all j. A is [M,K], DOut is [M,N].
// p -> i -> j order keeps DB[p,:] hot and sums over i ascending (the TS backward's order).
void MatMulTnScalar(const double *A, const double *DOut, double *DB,
                    int PStart, int PEnd, int M, int K, int N, int Acc) {
    for (int P = PStart; P < PEnd; P++) {
        double *Db = DB + (size_t)P * N;
        if (!Acc) {
            for (int J = 0; J < N; J++) Db[J] = 0.0;
        }
        for (int I = 0; I < M; I++) {
            double Av = A[(size_t)I * K + P];
            const double *Dr = DOut + (size_t)I * N;
            for (int J = 0; J < N; J++) Db[J] += Av * Dr[J];
        }
    }
}

// f32 twin of MatMulRowsScalar.
void MatMulRowsScalarF32(const float *A, const float *BT, float *O,
                         int RowStart, int RowEnd, int K, int N, int Acc) {
    for (int I = RowStart; I < RowEnd; I++) {
        const float *Ai = A + (size_t)I * K;
        float *Oi = O + (size_t)I * N;
        for (int J = 0; J < N; J++) {
            const float *Bt = BT + (size_t)J * K;
            float R = 0.0f;
            for (int P = 0; P < K; P++) R += Ai[P] * Bt[P];
            if (Acc) Oi[J] += R;
            else Oi[J] = R;
        }
    }
}

// f32 twin of MatMulTnScalar.
void MatMulTnScalarF32(const float *A, const float *DOut, float *DB,
                       int PStart, int PEnd, int M, int K, int N, int Acc) {
    for (int P = PStart; P < PEnd; P++) {
        float *Db = DB + (size_t)P * N;
        if (!Acc) {
            for (int J = 0; J < N; J++) Db[J] = 0.0f;
        }
        for (int I = 0; I < M; I++) {
            float Av = A[(size_t)I * K + P];
            const float *Dr = DOut + (size_t)I * N;
            for (int J = 0; J < N; J++) Db[J] += Av * Dr[J];
        }
    }
}
