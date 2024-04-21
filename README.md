# measure_bandwidth

Code allowing an approximate measurement of RAM bandwidth, within the context of vectorization (SSE, AVX2 or AVX-512). To check bandwidth for a particular vectorization scheme go to sse/ or avx2/ or avx512/, and execute 'sh run.sh'. Furthermore, go inside the script run.sh to switch between double/float as desired.

The following results are on a node Intel(R) Xeon(R) Platinum 8180 CPU.

-----------

These first results try to indirectly estimate the impact of RAM bandwidth in our computations. We run from sse/, avx2/, avx512/.

## SSE:

### float sse

Probing RAM bandwidth with SSE loading calls in float ...

 Setting random numbers ...

 ... done

 -- time elapsed : 3.1291080000 seconds

 -- data accessed: 31.471252 GB

 -- accessed data per second: 10.057579 GB/s

... done. Cleaning up now ...

... done

### double sse

Probing RAM bandwidth with SSE loading calls in double ...

 Setting random numbers ...

 ... done

 -- time elapsed : 7.5145750000 seconds

 -- data accessed: 62.942505 GB

 -- accessed data per second: 8.376057 GB/s

... done. Cleaning up now ...

... done


## AVX2:

Probing RAM bandwidth with AVX2 loading calls in float ...

 Setting random numbers ...

 ... done

 -- time elapsed : 2.6860060000 seconds

 -- data accessed: 31.471252 GB

 -- RAM memory bandwidth: 11.716747 GB/s

... done. Cleaning up now ...

... done


## AVX-512:

Probing RAM bandwidth with AVX512 loading calls in float ...

 Setting random numbers ...

 ... done

 -- time elapsed : 3.0182990000 seconds

 -- data accessed: 31.471252 GB

 -- RAM memory bandwidth: 10.426817 GB/s

... done. Cleaning up now ...

... done


## bare (basically only memory accesses)

### bare float

Probing RAM bandwidth loading calls in float ...

 Setting random numbers ...

 ... done

 -- time elapsed : 1.4163590000 seconds

 -- data accessed: 31.471252 GB

 -- accessed data per second: 22.219827 GB/s

... done. Cleaning up now ...

... done

### bare double

Probing RAM bandwidth loading calls in double ...

 Setting random numbers ...

 ... done

 -- time elapsed : 3.0817500000 seconds

 -- data accessed: 62.942505 GB

 -- accessed data per second: 20.424274 GB/s

... done. Cleaning up now ...

... done

-----------

The results that follow tell us about the impact of vectorization when doing the 'upgrades' SSE --> AVX2 --> AVX-512. We run from sse_pure_compute/, avx2_pure_compute/, 
avx512_pure_compute/.

## SSE:

### float sse

Probing 'pure compute' with SSE loading calls in float ...

 Setting random numbers ...

 ... done

 -- time elapsed : 1.5555160000 seconds

... done. Cleaning up now ...

... done

## AVX2:

Probing RAM bandwidth with AVX2 loading calls in float ...

 Setting random numbers ...

 ... done

 -- time elapsed : 0.5036230000 seconds

... done. Cleaning up now ...

... done

## AVX-512:

Probing RAM bandwidth with AVX512 loading calls in float ...

 Setting random numbers ...

 ... done

 -- time elapsed : 0.3161330000 seconds

... done. Cleaning up now ...

... done
