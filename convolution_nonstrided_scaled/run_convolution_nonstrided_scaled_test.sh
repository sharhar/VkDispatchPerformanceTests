#!/bin/bash

set -Eeuo pipefail
trap 'echo "Error in run_convolution_nonstrided_scaled_test.sh on line $LINENO: exit code $?"; exit 1' ERR

mkdir -p test_results

cd test_results

if [ -f "../../convolution_nvidia/test_results/nvidia_ratios.csv" ]; then
    cp ../../convolution_nvidia/test_results/nvidia_ratios.csv nvidia_ratios.csv
fi

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

$NVCC -O2 -std=c++17 ../cufft_test.cu -gencode arch=compute_${ARCH},code=sm_${ARCH} -rdc=true -lcufft_static -lculibos -o cufft_test.exec

echo "Running performance tests with the following parameters:"
echo "Data Size: $DATA_SIZE"
echo "Iteration Count: $ITER_COUNT"
echo "Batch Size: $BATCH_SIZE"
echo "Repeats: $REPEATS"

echo "Running cuFFT Test..."
./cufft_test.exec $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

echo "Running Vkdispatch Test..."
python3 ../vkdispatch_test.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

echo "Running PyTorch Test..."
python3 ../torch_test.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

echo "Running ZipFFT Test..."
python3 ../zipfft_test.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

python3 ../make_graph.py
python3 ../make_ratios_graph.py
