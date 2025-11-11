#!/bin/bash

mkdir -p test_results
cd test_results

if [ -n "${CUDA_HOME:-}" ]; then
    NVCC="$CUDA_HOME/bin/nvcc"
else
    NVCC="nvcc"
fi

DATA_SIZE=$1
ITER_COUNT=$2
BATCH_SIZE=$3
REPEATS=$4
ARCH=$5

echo "Running performance tests with the following parameters:"
echo "Data Size: $DATA_SIZE"
echo "Iteration Count: $ITER_COUNT"
echo "Batch Size: $BATCH_SIZE"
echo "Repeats: $REPEATS"

# echo "Running cuFFT FFT..."
# $NVCC -O2 -std=c++17 ../cufft_test.cu -gencode arch=compute_${ARCH},code=sm_${ARCH} -lcufft -lculibos -o cufft_test.exec
# ./cufft_test.exec $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS
# rm cufft_test.exec

# echo "Running Vkdispatch FFT..."
# python3 ../vkdispatch_test.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

echo "Running VKFFT FFT..."
python3 ../vkfft_test.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

# echo "Running PyTorch FFT..."
# python3 ../torch_test.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

# echo "Running ZipFFT FFT..."
# python3 ../zipfft_test.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS