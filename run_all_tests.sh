#! /bin/bash

mkdir -p test_results

#DATA_SIZE=134217728
DATA_SIZE=67108864
#DATA_SIZE=33554432
ITER_COUNT=100
BATCH_SIZE=10
REPEATS=3
ARCH=86

echo "Running all tests with DATA_SIZE=$DATA_SIZE, ITER_COUNT=$ITER_COUNT, BATCH_SIZE=$BATCH_SIZE, REPEATS=$REPEATS, ARCH=$ARCH"

echo "Running fft_nonstrided test..."
cd fft_nonstrided
bash run_fft_nonstrided_test.sh $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS $ARCH
cd ..
cp fft_nonstrided/test_results/fft_nonstrided_graph.png test_results/fft_nonstrided_graph.png

echo "Running convolution_nvidia test..."
cd convolution_nvidia
bash run_convolution_nvidia_test.sh $ARCH $REPEATS
cd ..
cp convolution_nvidia/test_results/convolution_nvidia_graph.png test_results/convolution_nvidia_graph.png

echo "Running convolution_nonstrided_scaled test..."
cd convolution_nonstrided_scaled
bash run_convolution_nonstrided_scaled_test.sh $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS $ARCH
cd ..
cp convolution_nonstrided_scaled/test_results/convolution_nonstrided_scaled_graph.png test_results/convolution_nonstrided_scaled_graph.png
cp convolution_nonstrided_scaled/test_results/convolution_nonstrided_scaled_ratios_graph.png test_results/convolution_nonstrided_scaled_ratios_graph.png