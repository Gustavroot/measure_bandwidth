/**
 * @file vectorized_blas_sse.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-02-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <immintrin.h>

#define SSE_LENGTH_float 4

// res = a*b + c
static inline __m128 sse_fmadd(__m128 a, __m128 b, __m128 c)
{
    __m128 res;
    res = _mm_mul_ps(a, b);
    res = _mm_add_ps(res, c);
    return res;
}


// res = a*b - c
static inline __m128 sse_fmsub(__m128 a, __m128 b, __m128 c)
{
    __m128 res;
    res = _mm_mul_ps(a, b);
    res = _mm_sub_ps(res, c);
    return res;
}


// c = a*b + c
static inline void cfmadd(__m128 a_real, __m128 a_imag, __m128 b_real, __m128 b_imag, __m128 *c_real, __m128 *c_imag)
{
    *c_real = sse_fmsub(a_imag, b_imag, *c_real);
    *c_imag = sse_fmadd(a_imag, b_real, *c_imag);
    *c_real = sse_fmsub(a_real, b_real, *c_real);
    *c_imag = sse_fmadd(a_real, b_imag, *c_imag);
}


// c = -a*b + c; a,b,c are SIMD variables, eg： a = a_real + i*a_imag
static inline void cfnmadd(__m128 a_real, __m128 a_imag, __m128 b_real, __m128 b_imag, __m128 *c_real, __m128 *c_imag)
{
    __m128 minus_a_real;
    __m128 minus_a_imag;
    minus_a_real = _mm_setzero_ps();
    minus_a_imag = _mm_sub_ps(minus_a_real, a_imag);
    minus_a_real = _mm_sub_ps(minus_a_real, a_real);

    *c_real = sse_fmsub(minus_a_imag, b_imag, *c_real);
    *c_imag = sse_fmadd(minus_a_imag, b_real, *c_imag);
    *c_real = sse_fmsub(minus_a_real, b_real, *c_real);
    *c_imag = sse_fmadd(minus_a_real, b_imag, *c_imag);
}



static inline void sse_cgemv(const int N, const float *A, const int lda, const float *B, float *C)
{
    // ProfSSE.load.start();
    int i, j;

    __m128 A_re;
    __m128 A_im;
    __m128 B_re;
    __m128 B_im;
    __m128 C_re[lda / SSE_LENGTH_float];
    __m128 C_im[lda / SSE_LENGTH_float];

    // deinterleaved load
    for (i = 0; i < lda; i += SSE_LENGTH_float) {
        C_re[i / SSE_LENGTH_float] = _mm_setr_ps(C[2 * i], C[2 * i + 2], C[2 * i + 4], C[2 * i + 6]);
        C_im[i / SSE_LENGTH_float] = _mm_setr_ps(C[2 * i + 1], C[2 * i + 3], C[2 * i + 5], C[2 * i + 7]);
    }
    // ProfSSE.load.stop();

    // ProfSSE.compute.start();
    for (j = 0; j < N; j++) {
        // load the j-th complex number in B
        B_re = _mm_set1_ps(B[2 * j]);
        B_im = _mm_set1_ps(B[2 * j + 1]);
        for (i = 0; i < lda; i += SSE_LENGTH_float) {
            A_re = _mm_load_ps(A + 2 * j * lda + i);
            A_im = _mm_load_ps(A + (2 * j + 1) * lda + i);

            // C += A*Bxzha
            cfmadd(A_re, A_im, B_re, B_im, &(C_re[i / SSE_LENGTH_float]), &(C_im[i / SSE_LENGTH_float]));
        }
    }
    // ProfSSE.compute.stop();

    // interleaves real and imaginary parts and stores the resulting complex numbers in C
    // ProfSSE.store.start();
    for (i = 0; i < lda; i += SSE_LENGTH_float) {
        A_re = _mm_unpacklo_ps(C_re[i / SSE_LENGTH_float], C_im[i / SSE_LENGTH_float]);
        A_im = _mm_unpackhi_ps(C_re[i / SSE_LENGTH_float], C_im[i / SSE_LENGTH_float]);
        _mm_store_ps(C + 2 * i, A_re);
        _mm_store_ps(C + 2 * i + SSE_LENGTH_float, A_im);
    }
    // ProfSSE.store.stop();
}


static inline void sse_cgenmv(const int N, const float *A, int lda, const float *B, float *C)
{
    // ProfSSE.load.start();
    int i, j;

    __m128 A_re;
    __m128 A_im;
    __m128 B_re;
    __m128 B_im;
    __m128 C_re[lda / SSE_LENGTH_float];
    __m128 C_im[lda / SSE_LENGTH_float];

    for (i = 0; i < lda; i += SSE_LENGTH_float) {
        C_re[i / SSE_LENGTH_float] = _mm_setr_ps(C[2 * i], C[2 * i + 2], C[2 * i + 4], C[2 * i + 6]);
        C_im[i / SSE_LENGTH_float] = _mm_setr_ps(C[2 * i + 1], C[2 * i + 3], C[2 * i + 5], C[2 * i + 7]);
    }
    // ProfSSE.load.stop();

    // ProfSSE.compute.start();
    for (j = 0; j < N; j++) {

        B_re = _mm_set1_ps(B[2 * j]);
        B_im = _mm_set1_ps(B[2 * j + 1]);

        for (i = 0; i < lda; i += SSE_LENGTH_float) {
            A_re = _mm_load_ps(A + 2 * j * lda + i);
            A_im = _mm_load_ps(A + (2 * j + 1) * lda + i);

            // C -= A*B
            cfnmadd(A_re, A_im, B_re, B_im, &(C_re[i / SSE_LENGTH_float]), &(C_im[i / SSE_LENGTH_float]));
        }
    }
    // ProfSSE.compute.stop();

    // ProfSSE.store.start();
    for (i = 0; i < lda; i += SSE_LENGTH_float) {
        A_re = _mm_unpacklo_ps(C_re[i / SSE_LENGTH_float], C_im[i / SSE_LENGTH_float]);
        A_im = _mm_unpackhi_ps(C_re[i / SSE_LENGTH_float], C_im[i / SSE_LENGTH_float]);
        _mm_store_ps(C + 2 * i, A_re);
        _mm_store_ps(C + 2 * i + SSE_LENGTH_float, A_im);
    }
    // ProfSSE.store.stop();
}
