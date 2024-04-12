#!/bin/bash


# to run this script just do : sh run.sh


PRECISION="float"
echo "Probing RAM bandwidth with AVX2 loading calls in $PRECISION ..."
cp main.c main_$PRECISION.c
SUB_STRING=`echo "s/PRECISION/$PRECISION/g"`
sed -i -e $SUB_STRING main_$PRECISION.c
gcc main_$PRECISION.c -mavx -mavx2 -mfma -DTYPE$PRECISION
taskset -c 0 ./a.out
echo "... done. Cleaning up now ..."

# cleaning up
rm main_$PRECISION.c
rm a.out
echo "... done"

echo "Probing RAM bandwidth with AVX2 loading calls in $PRECISION  With Computation..."
cp main.c main_$PRECISION.c
SUB_STRING=`echo "s/PRECISION/$PRECISION/g"`
sed -i -e $SUB_STRING main_$PRECISION.c
gcc main_$PRECISION.c -mfma -mavx -mavx2 -DTYPE$PRECISION -DWITH_COMP
taskset -c 0 ./a.out
echo "... done. Cleaning up now ..."

# cleaning up
rm main_$PRECISION.c
rm a.out
echo "... done"
