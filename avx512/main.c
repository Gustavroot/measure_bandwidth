#include <stdio.h>
#include <immintrin.h>
#include <sys/time.h>

#define AVX512_LENGTH_float  16

void block_mem_loader(const int N, const PRECISION *A, int lda, const PRECISION *B, PRECISION *C)
{
    __m512i idxe = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30); // addr of bytes
    __m512i idxo = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);

    const int Nsm = lda / AVX512_LENGTH_PRECISION;
    __m512 A_re;
    __m512 A_im;
    __m512 B_re;
    __m512 B_im;
    __m512 C_re[Nsm];
    __m512 C_im[Nsm];
    for (int i = 0; i < Nsm; i++) {
        C_re[i] = _mm512_i32gather_ps(idxe, C + i * 2 * AVX512_LENGTH_PRECISION, 4); //idxe * 4 bytes
        C_im[i] = _mm512_i32gather_ps(idxo, C + i * 2 * AVX512_LENGTH_PRECISION, 4);
    }
    for (int j = 0; j < N; j++) {
        B_re = _mm512_set1_ps(B[2 * j]);
        B_im = _mm512_set1_ps(B[2 * j + 1]);
        for (int i = 0; i < Nsm; i++) {
            A_re = _mm512_loadu_ps(A + 2 * j * lda + i * AVX512_LENGTH_PRECISION);
            A_im = _mm512_loadu_ps(A + (2 * j + 1) * lda + i * AVX512_LENGTH_PRECISION);

            // C += A*B
            C_re[i] = _mm512_fmsub_ps(A_re, B_re, _mm512_fmsub_ps(A_im, B_im, C_re[i]));
            C_im[i] = _mm512_fmadd_ps(A_im, B_re, _mm512_fmadd_ps(A_re, B_im, C_im[i]));
        }
    }
    for (int i = 0; i < lda; i += AVX512_LENGTH_PRECISION) {
        _mm512_i32scatter_ps(&C[2 * i], idxe, C_re[i / AVX512_LENGTH_PRECISION], 4);
        _mm512_i32scatter_ps(&C[2 * i], idxo, C_im[i / AVX512_LENGTH_PRECISION], 4);
    }
}

int main()
{
    int i,j,nr_sweeps=10;
    // dimension of the block
    int N  = 64;
    // number of blocks
    int nb = 1E5;

    PRECISION *Apt, *Bpt, *Cpt;

    struct timeval before = {};
    struct timeval after  = {};
    struct timeval result = {};

    PRECISION *A = (PRECISION *) malloc(nb * N * 2 * N * sizeof(PRECISION));
    PRECISION *B = (PRECISION *) malloc(nb * 2 * N * sizeof(PRECISION));
    PRECISION *C = (PRECISION *) malloc(nb * 2 * N * sizeof(PRECISION));

    printf(" Setting random numbers ...\n");
    for ( i=0;i<nb*N*2*N;i++ ) { A[i] = (PRECISION)rand()/(PRECISION)RAND_MAX; }
    for ( i=0;i<nb*2*N;i++ ) { B[i] = (PRECISION)rand()/(PRECISION)RAND_MAX; }
    for ( i=0;i<nb*2*N;i++ ) { C[i] = (PRECISION)rand()/(PRECISION)RAND_MAX; }
    printf(" ... done\n");

    gettimeofday(&before, NULL);
    for ( j=0;j<nr_sweeps;j++ ) {
        for (i = 0; i < nb; i++) {
            Apt = A + i * N * 2 * N;
            Bpt = B + i * 2 * N;
            Cpt = C + i * 2 * N;
            block_mem_loader(N, Apt, N, Bpt, Cpt);
        }
    }
    gettimeofday(&after, NULL);
    timersub(&after, &before, &result);

    double elaps_time = result.tv_sec + result.tv_usec * 1.0e-6;
    double data_size  = nr_sweeps * sizeof(PRECISION) * nb * (N * 2 * N + 2 * N + 2 * N) / 1024.0 / 1024.0 / 1024.0;
    printf(" -- time elapsed : %.10f seconds\n", elaps_time);
    printf(" -- data accessed: %f GB\n", data_size);
    printf(" -- RAM memory bandwidth: %f GB/s\n", data_size / elaps_time);

    free(A);
    free(B);
    free(C);
}
