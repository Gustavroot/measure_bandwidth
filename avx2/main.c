#include <stdio.h>
#include <immintrin.h>
#include <sys/time.h>

#define AVX_LENGTH_float  8
#define AVX_LENGTH_double 4

// TODO
// #define _mm_setr_pfloat      _mm_setr_ps
// #define _mm_setr_pdouble     _mm_setr_pd
// #define _mm_set1_pfloat      _mm_set1_ps
// #define _mm_set1_pdouble     _mm_set1_pd
// #define _mm_load_pfloat      _mm_load_ps
// #define _mm_load_pdouble     _mm_load_pd
// #define _mm_unpacklo_pfloat  _mm_unpacklo_ps
// #define _mm_unpacklo_pdouble _mm_unpacklo_pd
// #define _mm_unpackhi_pfloat  _mm_unpackhi_ps
// #define _mm_unpackhi_pdouble _mm_unpackhi_pd
// #define _mm_store_pfloat     _mm_store_ps
// #define _mm_store_pdouble    _mm_store_pd

#define type_sse_reg_float  __m256
#define type_sse_reg_double __m256d

void block_mem_loader(const int N, const PRECISION *A, int lda, const PRECISION *B, PRECISION *C)
{
    __m256i idxe = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14); // addr of bytes
    __m256i idxo = _mm256_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15);

    __m256 A_re;
    __m256 A_im;
    __m256 B_re;
    __m256 B_im;
    __m256 C_re[lda / AVX_LENGTH_float];
    __m256 C_im[lda / AVX_LENGTH_float];

    for (int i = 0; i < lda; i += AVX_LENGTH_float) {
        C_re[i / AVX_LENGTH_float] = _mm256_i32gather_ps(&C[2 * i], idxe, 4); // idxe * 4 bytes
        C_im[i / AVX_LENGTH_float] = _mm256_i32gather_ps(&C[2 * i], idxo, 4);
    }

    for (int j = 0; j < N; j++) {
        B_re = _mm256_set1_ps(B[2 * j]);
        B_im = _mm256_set1_ps(B[2 * j + 1]);
        for (int i = 0; i < lda; i += AVX_LENGTH_float) {
            A_re = _mm256_loadu_ps(A + 2 * j * lda + i);
            A_im = _mm256_loadu_ps(A + (2 * j + 1) * lda + i);
#ifdef WITH_COMP
            C_re[i / AVX_LENGTH_float] = _mm256_fmsub_ps(A_re, B_re, _mm256_fmsub_ps(A_im, B_im, C_re[i / AVX_LENGTH_float]));
            C_im[i / AVX_LENGTH_float] = _mm256_fmadd_ps(A_im, B_re, _mm256_fmadd_ps(A_re, B_im, C_im[i / AVX_LENGTH_float]));
#endif
        }
    }

    __m256i idxA = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    for (int i = 0; i < lda; i += AVX_LENGTH_float) {
        __m256 A_re, A_im;
        __m256 B_re, B_im;
        A_re = _mm256_permutevar8x32_ps(C_re[i / AVX_LENGTH_float], idxA);
        A_im = _mm256_permutevar8x32_ps(C_im[i / AVX_LENGTH_float], idxA);
        B_re = _mm256_unpacklo_ps(A_re, A_im);
        B_im = _mm256_unpackhi_ps(A_re, A_im);
        _mm256_storeu_ps(C + 2 * i, B_re);
        _mm256_storeu_ps(C + 2 * i + AVX_LENGTH_float, B_im);
    }
}

int main()
{

    int i;
    // dimension of the block
    int N  = 32;
    // number of blocks
    int nb = 1E6 / sizeof(PRECISION);

    PRECISION *Apt, *Bpt, *Cpt;

    struct timeval before = {};
    struct timeval after  = {};
    struct timeval result = {};

    PRECISION *A = (PRECISION *) malloc(nb * N * 2 * N * sizeof(PRECISION));
    PRECISION *B = (PRECISION *) malloc(nb * 2 * N * sizeof(PRECISION));
    PRECISION *C = (PRECISION *) malloc(nb * 2 * N * sizeof(PRECISION));

    gettimeofday(&before, NULL);

    for (i = 0; i < nb; i++) {
        Apt = A + i * N * 2 * N;
        Bpt = B + i * 2 * N;
        Cpt = C + i * 2 * N;
        block_mem_loader(N, Apt, N, Bpt, Cpt);
    }

    gettimeofday(&after, NULL);
    timersub(&after, &before, &result);
    double elaps_time = result.tv_sec + result.tv_usec * 1.0e-6;
    printf(" -- time elapsed : %.10f seconds\n", elaps_time);
    double data_size = sizeof(PRECISION) * nb * (N * 2 * N + 2 * N + 2 * N) / 1024.0 / 1024.0 / 1024.0;
    printf(" -- data accessed: %f GB\n", data_size);
    printf(" -- RAM memory bandwidth: %f GB/s\n", data_size / elaps_time);

    free(A);
    free(B);
    free(C);
}
