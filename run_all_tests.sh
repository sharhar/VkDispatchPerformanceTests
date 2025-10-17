#! /bin/bash

#DATA_SIZE=134217728
DATA_SIZE=67108864
#DATA_SIZE=33554432
ITER_COUNT=100
BATCH_SIZE=10
REPEATS=3
ARCH=86

cd convolution_nonstrided_scaled
bash run_convolution_nonstrided_scaled_test.sh $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS $ARCH
cd ..