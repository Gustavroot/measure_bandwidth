#include <stdio.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <sys/time.h>



#define SIMD_LENGTH_float 4
#define SIMD_LENGTH_double 2

#define _mm_setr_pfloat _mm_setr_ps
#define _mm_setr_pdouble _mm_setr_pd
#define _mm_set1_pfloat _mm_set1_ps
#define _mm_set1_pdouble _mm_set1_pd
#define _mm_load_pfloat _mm_load_ps
#define _mm_load_pdouble _mm_load_pd
#define _mm_unpacklo_pfloat _mm_unpacklo_ps
#define _mm_unpacklo_pdouble _mm_unpacklo_pd
#define _mm_unpackhi_pfloat _mm_unpackhi_ps
#define _mm_unpackhi_pdouble _mm_unpackhi_pd
#define _mm_store_pfloat _mm_store_ps
#define _mm_store_pdouble _mm_store_pd


#define type_sse_reg_float __m128
#define type_sse_reg_double __m128d



void block_mem_loader( const int N, const PRECISION *A, int lda, const PRECISION *B, PRECISION *C ) {
  int i, j;

  type_sse_reg_PRECISION A_re;
  type_sse_reg_PRECISION A_im;
  type_sse_reg_PRECISION B_re;
  type_sse_reg_PRECISION B_im;
  type_sse_reg_PRECISION C_re[lda/SIMD_LENGTH_PRECISION];
  type_sse_reg_PRECISION C_im[lda/SIMD_LENGTH_PRECISION];

  // deinterleaved load
  for ( i=0; i<lda; i+= SIMD_LENGTH_PRECISION ) {
#ifdef TYPEfloat
    C_re[i/SIMD_LENGTH_PRECISION] = _mm_setr_pPRECISION(C[2*i], C[2*i+2], C[2*i+4], C[2*i+6] );
    C_im[i/SIMD_LENGTH_PRECISION] = _mm_setr_pPRECISION(C[2*i+1], C[2*i+3], C[2*i+5], C[2*i+7] );
#else
    C_re[i/SIMD_LENGTH_PRECISION] = _mm_setr_pPRECISION(C[2*i], C[2*i+2] );
    C_im[i/SIMD_LENGTH_PRECISION] = _mm_setr_pPRECISION(C[2*i+1], C[2*i+3] );
#endif
  }

  for ( j=0; j<N; j++ ) {
    // load the j-th complex number in B
    B_re = _mm_set1_pPRECISION( B[2*j] );
    B_im = _mm_set1_pPRECISION( B[2*j+1] );

    for ( i=0; i<lda; i+= SIMD_LENGTH_PRECISION ) {
       A_re = _mm_load_pPRECISION( A + 2*j*lda + i );
       A_im = _mm_load_pPRECISION( A + (2*j+1)*lda + i );

       // C += A*B
       //cfmadd(A_re, A_im, B_re, B_im, &(C_re[i/SIMD_LENGTH_PRECISION]), &(C_im[i/SIMD_LENGTH_PRECISION]) );
    }
  }

  // interleaves real and imaginary parts and stores the resulting complex numbers in C
  for ( i=0; i<lda; i+= SIMD_LENGTH_PRECISION ) {
     A_re = _mm_unpacklo_pPRECISION( C_re[i/SIMD_LENGTH_PRECISION], C_im[i/SIMD_LENGTH_PRECISION] );
     A_im = _mm_unpackhi_pPRECISION( C_re[i/SIMD_LENGTH_PRECISION], C_im[i/SIMD_LENGTH_PRECISION] );
     _mm_store_pPRECISION( C+2*i,                   A_re );
     _mm_store_pPRECISION( C+2*i+SIMD_LENGTH_PRECISION, A_im );
  }
}



int main(){

  int i;
  // dimension of the block
  int N = 32;
  // number of blocks
  int nb = 1E6/sizeof(PRECISION);

  PRECISION *Apt,*Bpt,*Cpt;

  struct timeval before = {};
  struct timeval after = {}; 
  struct timeval result = {};

  PRECISION *A = (PRECISION *) malloc( nb*N*2*N*sizeof(PRECISION) );
  PRECISION *B = (PRECISION *) malloc( nb*2*N*sizeof(PRECISION) );
  PRECISION *C = (PRECISION *) malloc( nb*2*N*sizeof(PRECISION) );

  gettimeofday(&before, NULL);

  for ( i=0;i<nb;i++ ) {
    Apt = A+i*N*2*N; Bpt = B+i*2*N; Cpt = C+i*2*N;
    block_mem_loader( N,Apt,N,Bpt,Cpt );
  }

  gettimeofday(&after, NULL);
  timersub(&after, &before, &result);
  double elaps_time = result.tv_sec+result.tv_usec*1.0e-6;
  printf(" -- time elapsed : %.10f seconds\n", elaps_time);
  double data_size = sizeof(PRECISION)*nb*(N*2*N+2*N+2*N)/1024.0/1024.0/1024.0;
  printf(" -- data accessed: %f GB\n", data_size);
  printf(" -- RAM memory bandwidth: %f GB/s\n", data_size/elaps_time);

  free(A);
  free(B);
  free(C);
}
