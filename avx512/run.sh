#!/bin/bash


# to run this script just do : sh run.sh


PRECISION="float"
if [ "$PRECISION" != "float" ]
then
  echo "This bandwidth extraction with AVX-512 is supporting float only at the moment"
  exit 1
fi
echo "Probing RAM bandwidth with AVX512 loading calls in $PRECISION ..."
cp main.c main_$PRECISION.c
SUB_STRING=`echo "s/PRECISION/$PRECISION/g"`
sed -i -e $SUB_STRING main_$PRECISION.c
gcc main_$PRECISION.c -mavx512vl -mavx512f -mfma -DTYPE$PRECISION
taskset -c 0 ./a.out
echo "... done. Cleaning up now ..."

# cleaning up
rm main_$PRECISION.c
rm a.out
echo "... done"

echo "Probing RAM bandwidth with AVX512 loading calls in $PRECISION  With Computation..."
cp main.c main_$PRECISION.c
SUB_STRING=`echo "s/PRECISION/$PRECISION/g"`
sed -i -e $SUB_STRING main_$PRECISION.c
gcc main_$PRECISION.c -mfma -mavx512vl -mavx512vl -DTYPE$PRECISION -DWITH_COMP
taskset -c 0 ./a.out
echo "... done. Cleaning up now ..."

# cleaning up
rm main_$PRECISION.c
rm a.out
echo "... done"

