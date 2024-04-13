#include <stdio.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <sys/time.h>
#include <immintrin.h>

// this is the number of either float or double fitting in
// a cache line, which we assume here to be 64 bytes
#define CHNK_SIZE_float 16
#define CHNK_SIZE_double 8

int main(){

  int i,j,ix,jx,nr_sweeps=10;
  // dimension of the block
  int N = 64;
  // number of blocks
  int nb = 1E5;

  struct timeval before = {};
  struct timeval after = {}; 
  struct timeval result = {};

  PRECISION *A = (PRECISION *) malloc( nb*N*2*N*sizeof(PRECISION) );
  PRECISION *B = (PRECISION *) malloc( nb*2*N*sizeof(PRECISION) );
  PRECISION *C = (PRECISION *) malloc( nb*2*N*sizeof(PRECISION) );

  printf(" Setting random numbers ...\n");
  for ( i=0;i<nb*N*2*N;i++ ) { A[i] = (PRECISION)rand()/(PRECISION)RAND_MAX; }
  for ( i=0;i<nb*2*N;i++ ) { B[i] = (PRECISION)rand()/(PRECISION)RAND_MAX; }
  for ( i=0;i<nb*2*N;i++ ) { C[i] = (PRECISION)rand()/(PRECISION)RAND_MAX; }
  printf(" ... done\n");

  gettimeofday(&before, NULL);
  for ( j=0;j<nr_sweeps;j++ ) {
    for ( i=0;i<nb;i++ ) {
      for ( ix=0;ix<2*N;ix++ ) { C[ix] = 0.0; }
      for ( jx=0;jx<2*N;jx++ ) {
        for ( ix=0;ix<N;ix++ ) {
          C[ix]   += A[jx*N+ix] + B[ix];
          C[ix+N] += A[jx*N+ix] + B[ix+N];
        }
      }
    }
  }
  gettimeofday(&after, NULL);
  timersub(&after, &before, &result);

  double elaps_time = result.tv_sec+result.tv_usec*1.0e-6;
  double data_size = nr_sweeps*sizeof(PRECISION)*nb*(N*2*N+2*N+2*N)/1024.0/1024.0/1024.0;
  printf(" -- time elapsed : %.10f seconds\n", elaps_time);
  printf(" -- data accessed: %f GB\n", data_size);
  printf(" -- accessed data per second: %f GB/s\n", data_size/elaps_time);

  free(A);
  free(B);
  free(C);
}
