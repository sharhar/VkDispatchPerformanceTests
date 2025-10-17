#! /bin/bash

mkdir -p test_results

#cd test_results

#DATA_SIZE=134217728
DATA_SIZE=67108864
#DATA_SIZE=33554432
ITER_COUNT=100
BATCH_SIZE=10
REPEATS=3
ARCH=86

# cd fft_nonstrided
# bash run_fft_nonstrided_test.sh $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS $ARCH
# cd ..
cp fft_nonstrided/test_results/fft_nonstrided_graph.png test_results/fft_nonstrided_graph.png

#cd convolution_nonstrided_scaled
#bash run_convolution_nonstrided_scaled_test.sh $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS $ARCH
#cd ..
cp convolution_nonstrided_scaled/test_results/convolution_nonstrided_scaled_graph.png test_results/convolution_nonstrided_scaled_graph.png
cp convolution_nonstrided_scaled/test_results/convolution_nonstrided_scaled_ratios_graph.png test_results/convolution_nonstrided_scaled_ratios_graph.png