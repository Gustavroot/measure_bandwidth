# measure_bandwidth

Code allowing an approximate measurement of RAM bandwidth, within the context of vectorization (SSE, AVX2 or AVX-512). To check bandwidth for a particular vectorization scheme go to sse/ or avx2/ or avx512/, and execute 'sh run.sh'. Furthermore, go inside the script run.sh to switch between double/float as desired.

-----------

The following results are on a node Intel(R) Xeon(R) Platinum 8180 CPU.


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
