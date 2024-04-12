/**
 * @file benchmark_vectorized_blas_avx.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-01-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <immintrin.h>
#include "profiling.h"

#define AVX512_LENGTH_float 16
#define SSE_LENGTH_float    4
#define RELEASE
#if 0
static inline void avx512_cgemv(const int N, const float *A, int lda, const float *B, float *C)
{

    float ret[2 * lda];
    for (int i = 0; i < 2 * lda; i++) { ret[i] = C[i]; }
    for (int i = 0; i < lda; i++) {
        for (int j = 0; j < N; j++) {
            ret[2 * i + 0] += A[(2 * j + 0) * lda + i] * B[2 * j + 0] - A[(2 * j + 1) * lda + i] * B[2 * j + 1];
            ret[2 * i + 1] += A[(2 * j + 1) * lda + i] * B[2 * j + 0] + A[(2 * j + 0) * lda + i] * B[2 * j + 1];
        }
    }
    for (int i = 0; i < 2 * lda; i++) { C[i] = ret[i]; }


    // const int Nsm = lda / AVX512_LENGTH_float;

    // for (int i = 0; i < Nsm; i++) {
    //     // Prof512.load.start();
    //     __m512 C_re = _mm512_loadu_ps(C + 2 * i * AVX512_LENGTH_float); //idxe * 4 bytes
    //     __m512 C_im = _mm512_loadu_ps(C + (2 * i + 1) * AVX512_LENGTH_float);
    //     // Prof512.load.stop();

    //     // Prof512.compute.start();
    //     for (int j = 0; j < N; j++) {
    //         __m512 B_re = _mm512_set1_ps(B[2 * j]);
    //         __m512 B_im = _mm512_set1_ps(B[2 * j + 1]);
    //         __m512 A_re = _mm512_load_ps(A + 2 * j * lda + i * AVX512_LENGTH_float);
    //         __m512 A_im = _mm512_load_ps(A + (2 * j + 1) * lda + i * AVX512_LENGTH_float);
    //         C_re        = _mm512_fmsub_ps(A_re, B_re, _mm512_fmsub_ps(A_im, B_im, C_re));
    //         C_im        = _mm512_fmadd_ps(A_im, B_re, _mm512_fmadd_ps(A_re, B_im, C_im));
    //     }
    //     // Prof512.compute.stop();

    //     // store to float *C
    //     // Prof512.store.start();
    //     _mm512_storeu_ps(C + 2 * i * AVX512_LENGTH_float, C_re);
    //     _mm512_storeu_ps(C + (2 * i + 1) * AVX512_LENGTH_float, C_im);
    // }

#if 0
    const int Nsm = lda / AVX512_LENGTH_float;
    __m512i idxe  = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30); // addr of bytes
    __m512i idxo  = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);

    for (int i = 0; i < Nsm; i++) {
        __m512 C_re   = _mm512_i32gather_ps(idxe, C + i * 2 * AVX512_LENGTH_float, 4); //idxe * 4 bytes
        __m512 C_im   = _mm512_i32gather_ps(idxo, C + i * 2 * AVX512_LENGTH_float, 4);
        __m512 tmp_re = _mm512_setzero_ps();
        __m512 tmp_im = _mm512_setzero_ps();

        for (int j = 0; j < N; j += 2) {
            __m512 A_re  = _mm512_load_ps(A + 2 * j * lda + i * AVX512_LENGTH_float);
            __m512 A_im  = _mm512_load_ps(A + (2 * j + 1) * lda + i * AVX512_LENGTH_float);
            __m512 A_re1 = _mm512_load_ps(A + (2 * j + 2) * lda + i * AVX512_LENGTH_float);
            __m512 A_im1 = _mm512_load_ps(A + (2 * j + 3) * lda + i * AVX512_LENGTH_float);
            __m512 B_re  = _mm512_set1_ps(B[2 * j]);
            __m512 B_im  = _mm512_set1_ps(B[2 * j + 1]);
            C_re         = _mm512_fmsub_ps(A_re, B_re, _mm512_fmsub_ps(A_im, B_im, C_re));
            C_im         = _mm512_fmadd_ps(A_im, B_re, _mm512_fmadd_ps(A_re, B_im, C_im));
            __m512 B_re1 = _mm512_set1_ps(B[2 * j + 2]);
            __m512 B_im1 = _mm512_set1_ps(B[2 * j + 3]);


            tmp_re = _mm512_fmsub_ps(A_re1, B_re1, _mm512_fmsub_ps(A_im1, B_im1, tmp_re));
            tmp_im = _mm512_fmadd_ps(A_im1, B_re1, _mm512_fmadd_ps(A_re1, B_im1, tmp_im));
        }
        tmp_re = _mm512_add_ps(C_re, tmp_re);
        _mm512_i32scatter_ps(C + 2 * i * AVX512_LENGTH_float, idxe, tmp_re, 4);
        tmp_im = _mm512_add_ps(C_im, tmp_im);
        _mm512_i32scatter_ps(C + 2 * i * AVX512_LENGTH_float, idxo, tmp_im, 4);
    }
#endif

// deinterleaved load
#if 0
    // Prof512.load.start();
    for (int i = 0; i < Nsm; i++) {
        // Prof512.load.start();
        __m512 C_re = _mm512_i32gather_ps(idxe, C + i * 2 * AVX512_LENGTH_float, 4); //idxe * 4 bytes
        __m512 C_im = _mm512_i32gather_ps(idxo, C + i * 2 * AVX512_LENGTH_float, 4);
        // Prof512.load.stop();

        // Prof512.compute.start();
        for (int j = 0; j < N; j++) {
            __m512 B_re = _mm512_set1_ps(B[2 * j]);
            __m512 B_im = _mm512_set1_ps(B[2 * j + 1]);
            __m512 A_re = _mm512_load_ps(A + 2 * j * lda + i * AVX512_LENGTH_float);
            __m512 A_im = _mm512_load_ps(A + (2 * j + 1) * lda + i * AVX512_LENGTH_float);
            C_re        = _mm512_fmsub_ps(A_re, B_re, _mm512_fmsub_ps(A_im, B_im, C_re));
            C_im        = _mm512_fmadd_ps(A_im, B_re, _mm512_fmadd_ps(A_re, B_im, C_im));
        }
        // Prof512.compute.stop();

        // store to float *C
        // Prof512.store.start();
        _mm512_i32scatter_ps(C + 2 * i * AVX512_LENGTH_float, idxe, C_re, 4);
        _mm512_i32scatter_ps(C + 2 * i * AVX512_LENGTH_float, idxo, C_im, 4);
        // Prof512.store.stop();
    }
    // Prof512.load.stop();
#endif
}


// inline loadVectorA(__m512 *A_re, __m512 *A_im) {}

// inline void macVector(__m512 *C_re, __m512 *C_im, __m512 re, __m512 im, __m512 *A_re, __m512 *A_im, const int Nsm)
// {
//     C_re[i] = _mm512_fmsub_ps(A_re[i], B_re, _mm512_fmsub_ps(A_im[i], B_im, C_re[i]));
//     C_im[i] = _mm512_fmadd_ps(A_im[i], B_re, _mm512_fmadd_ps(A_re[i], B_im, C_im[i]));
// }
#else
static inline void avx512_cgemv(const int N, const float *A, int lda, const float *B, float *C)
{
#if defined(RELEASE)

    __m512i idxe = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30); // addr of bytes
    __m512i idxo = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);

    const int Nsm = lda / AVX512_LENGTH_float;

    __m512 A_re;
    __m512 A_im;
    __m512 B_re;
    __m512 B_im;
    __m512 C_re[Nsm];
    __m512 C_im[Nsm];

    // Prof512.reset();
    // deinterleaved load
    for (int i = 0; i < Nsm; i++) {
        C_re[i] = _mm512_i32gather_ps(idxe, C + i * 2 * AVX512_LENGTH_float, 4); //idxe * 4 bytes
        C_im[i] = _mm512_i32gather_ps(idxo, C + i * 2 * AVX512_LENGTH_float, 4);
    }

    // timeLoadAVX += Prof512.use_usec();

    // Prof512.reset();
    for (int j = 0; j < N; j++) {
        B_re = _mm512_set1_ps(B[2 * j]);
        B_im = _mm512_set1_ps(B[2 * j + 1]);
        for (int i = 0; i < Nsm; i++) {
            A_re    = _mm512_load_ps(A + 2 * j * lda + i * AVX512_LENGTH_float);
            A_im    = _mm512_load_ps(A + (2 * j + 1) * lda + i * AVX512_LENGTH_float);
            // C += A*B
            C_re[i] = _mm512_fmsub_ps(A_re, B_re, _mm512_fmsub_ps(A_im, B_im, C_re[i]));
            C_im[i] = _mm512_fmadd_ps(A_im, B_re, _mm512_fmadd_ps(A_re, B_im, C_im[i]));

            // C_re[i] = _mm512_fmsub_ps(A_re, B_re, _mm512_fmsub_ps(A_im, B_im, C_re[i]));
            // C_im[i] = _mm512_fmadd_ps(A_im, B_re, _mm512_fmadd_ps(A_re, B_im, C_im[i]));
        }
    }
    // timeComputAVX += Prof512.use_usec();

    // store to float *C
    for (int i = 0; i < lda; i += AVX512_LENGTH_float) {
        _mm512_i32scatter_ps(&C[2 * i], idxe, C_re[i / AVX512_LENGTH_float], 4);
        _mm512_i32scatter_ps(&C[2 * i], idxo, C_im[i / AVX512_LENGTH_float], 4);
    }
#elif defined(DEVEL)
    // Prof512.reset();
    __m512 A_re[Nsm];
    __m512 A_im[Nsm];
    float *pAre, *pAim;
    for (int j = 0; j < N; j++) {
        // Prof512.reset();
        B_re = _mm512_set1_ps(B[2 * j]);
        B_im = _mm512_set1_ps(B[2 * j + 1]);
        // timeLoadAVX += Prof512.use_usec();
        // Prof512.reset();
        pAre = (float *) &A[2 * j * lda];
        pAim = (float *) &A[(2 * j + 1) * lda];
        for (int i = 0; i < Nsm; i++) { A_re[i] = _mm512_loadu_ps(pAre + i * AVX512_LENGTH_float); }
        for (int i = 0; i < Nsm; i++) { A_im[i] = _mm512_loadu_ps(pAim + i * AVX512_LENGTH_float); }

        // timeLoadAVX += Prof512.use_usec();

        // Prof512.reset();
        for (int i = 0; i < Nsm; i++) {
            // C += A*B
            C_re[i] = _mm512_fmsub_ps(A_re[i], B_re, _mm512_fmsub_ps(A_im[i], B_im, C_re[i]));
            C_im[i] = _mm512_fmadd_ps(A_im[i], B_re, _mm512_fmadd_ps(A_re[i], B_im, C_im[i]));
        }
        // timeComputAVX += Prof512.use_usec();
    }

    // store to float *C
    for (int i = 0; i < lda / AVX512_LENGTH_float; i++) {
        float *pC = C + i * 2 * AVX512_LENGTH_float;
        __m128 re, im;
        re = _mm512_extractf32x4_ps(C_re[i], 0b00);
        im = _mm512_extractf32x4_ps(C_im[i], 0b00);
        _mm_store_ps(pC, _mm_unpacklo_ps(re, im));
        _mm_store_ps(pC + SSE_LENGTH_float, _mm_unpackhi_ps(re, im));

        pC = pC + 2 * SSE_LENGTH_float;
        re = _mm512_extractf32x4_ps(C_re[i], 0b01);
        im = _mm512_extractf32x4_ps(C_im[i], 0b01);
        _mm_store_ps(pC, _mm_unpacklo_ps(re, im));
        _mm_store_ps(pC + SSE_LENGTH_float, _mm_unpackhi_ps(re, im));

        pC = pC + 2 * SSE_LENGTH_float;
        re = _mm512_extractf32x4_ps(C_re[i], 0b10);
        im = _mm512_extractf32x4_ps(C_im[i], 0b10);
        _mm_store_ps(pC, _mm_unpacklo_ps(re, im));
        _mm_store_ps(pC + SSE_LENGTH_float, _mm_unpackhi_ps(re, im));

        pC = pC + 2 * SSE_LENGTH_float;
        re = _mm512_extractf32x4_ps(C_re[i], 0b11);
        im = _mm512_extractf32x4_ps(C_im[i], 0b11);
        _mm_store_ps(pC, _mm_unpacklo_ps(re, im));
        _mm_store_ps(pC + SSE_LENGTH_float, _mm_unpackhi_ps(re, im));
    }
#endif
    // timeStoreAVX += Prof512.use_usec();
}
#endif

static inline void avx512_cgenmv(const int N, const float *A, int lda, const float *B, float *C)
{
    __m512i idxe = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30); // addr of bytes
    __m512i idxo = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);

    __m512 A_re;
    __m512 A_im;
    __m512 B_re;
    __m512 B_im;
    __m512 C_re[lda / AVX512_LENGTH_float];
    __m512 C_im[lda / AVX512_LENGTH_float];

    // deinterleaved load
    for (int i = 0; i < lda; i += AVX512_LENGTH_float) {
        C_re[i / AVX512_LENGTH_float] = _mm512_i32gather_ps(idxe, &C[2 * i], 4); //idxe * 4 bytes
        C_im[i / AVX512_LENGTH_float] = _mm512_i32gather_ps(idxo, &C[2 * i], 4);
    }

    for (int j = 0; j < N; j++) {
        B_re = _mm512_set1_ps(B[2 * j]);
        B_im = _mm512_set1_ps(B[2 * j + 1]);

        for (int i = 0; i < lda; i += AVX512_LENGTH_float) {
            A_re = _mm512_loadu_ps(A + 2 * j * lda + i);
            A_im = _mm512_loadu_ps(A + (2 * j + 1) * lda + i);

            // C -= A*B
            C_re[i / AVX512_LENGTH_float] =
                _mm512_fnmadd_ps(A_re, B_re, _mm512_fmadd_ps(A_im, B_im, C_re[i / AVX512_LENGTH_float]));
            C_im[i / AVX512_LENGTH_float] =
                _mm512_fnmadd_ps(A_re, B_im, _mm512_fnmadd_ps(A_im, B_re, C_im[i / AVX512_LENGTH_float]));
        }
    }

    // store to float *C
    for (int i = 0; i < lda; i += AVX512_LENGTH_float) {
        _mm512_i32scatter_ps(&C[2 * i], idxe, C_re[i / AVX512_LENGTH_float], 4);
        _mm512_i32scatter_ps(&C[2 * i], idxo, C_im[i / AVX512_LENGTH_float], 4);
    }
}