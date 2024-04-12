#!/bin/bash


# to run this script just do : sh run.sh


PRECISION="float"
echo "Probing RAM bandwidth with SSE loading calls in $PRECISION ..."
cp main.c main_$PRECISION.c
SUB_STRING=`echo "s/PRECISION/$PRECISION/g"`
sed -i -e $SUB_STRING main_$PRECISION.c
gcc main_$PRECISION.c -msse4.2 -msse -DTYPE$PRECISION
taskset -c 0 ./a.out
echo "... done. Cleaning up now ..."

# cleaning up
rm main_$PRECISION.c
rm a.out
echo "... done"

echo "Probing RAM bandwidth with SSE loading calls in $PRECISION  With Computation..."
cp main.c main_$PRECISION.c
SUB_STRING=`echo "s/PRECISION/$PRECISION/g"`
sed -i -e $SUB_STRING main_$PRECISION.c
gcc main_$PRECISION.c -msse4.2 -msse -DTYPE$PRECISION -DWITH_COMP
taskset -c 0 ./a.out
echo "... done. Cleaning up now ..."

# cleaning up
rm main_$PRECISION.c
rm a.out
echo "... done"
