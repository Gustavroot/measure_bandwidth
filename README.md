# measure_bandwidth

Code allowing an approximate measurement of RAM bandwidth, within the context of vectorization (SSE, AVX2 or AVX-512). To check bandwidth for a particular vectorization scheme go to sse/ or avx2/ or avx512/, and execute 'sh run.sh'. Furthermore, go inside the script run.sh to switch between double/float as desired.

-----------

The following results are on a node Intel(R) Xeon(R) Platinum 8180 CPU.


## SSE:

Probing RAM bandwidth with SSE loading calls in float ...

 -- time elapsed : 44.8552790000 seconds

 -- data accessed: 393.390656 GB

 -- RAM memory bandwidth: 8.770220 GB/s

... done. Cleaning up now ...

... done


## AVX2:

Probing RAM bandwidth with AVX2 loading calls in float ...

 -- time elapsed : 27.2393060000 seconds

 -- data accessed: 393.390656 GB

 -- RAM memory bandwidth: 14.442022 GB/s

... done. Cleaning up now ...

... done


## AVX-512:

Probing RAM bandwidth with AVX512 loading calls in float ...

 -- time elapsed : 21.0683500000 seconds

 -- data accessed: 393.390656 GB

 -- RAM memory bandwidth: 18.672115 GB/s

... done. Cleaning up now ...

... done

