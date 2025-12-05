#! /bin/bash

# Check if run_test.sh exists in current directory
if [ -f "run_test.sh" ]; then
    ./run_test.sh "$@"
    exit 0
fi

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
RUN_CUDA=$6


echo "Running performance tests with the following parameters:"
echo "Data Size: $DATA_SIZE"
echo "Iteration Count: $ITER_COUNT"
echo "Batch Size: $BATCH_SIZE"
echo "Repeats: $REPEATS"


if [[ "$RUN_CUDA" == "true" ]] || [[ "$RUN_CUDA" == "1" ]]; then
    if [ -f "../cufft_test.cu" ]; then
        echo "Compiling cuFFT Test..."
        $NVCC -O3 -std=c++17 ../cufft_test.cu -gencode arch=compute_${ARCH},code=sm_${ARCH} -lcufft -lculibos -o cufft_test.exec
        echo "Running cuFFT Test..."
        ./cufft_test.exec $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS
        rm cufft_test.exec
    else
        echo "Skipping cuFFT Test - ../cufft_test.cu not found"
    fi

    if [ -f "../cufftdx_test.cu" ]; then
        echo "Compiling cuFFTdx Test..."
        $NVCC ../cufftdx_test.cu \
                    -std=c++17 -O3 \
                    -I ../../../dependencies/cutlass/include \
                    -I ../../../dependencies/nvidia-mathdx-25.06.1/nvidia/mathdx/25.06/include \
                    -DFFTS_PER_BLOCK=4 \
                    -DARCH=${ARCH}0 \
                    -gencode arch=compute_${ARCH},code=sm_${ARCH} \
                    -lcufft -lculibos \
                    -o cufftdx_test.exec
        echo "Running cuFFTdx Test..."
        ./cufftdx_test.exec $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS
        rm cufftdx_test.exec
    else
        echo "Skipping cuFFTdx Test - ../cufftdx_test.cu not found"
    fi
fi

if [ -f "../vkdispatch_test.py" ]; then
    echo "Running Vkdispatch Test..."
    python3 ../vkdispatch_test.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS
else
    echo "Skipping Vkdispatch Test - ../vkdispatch_test.py not found"
fi

if [ -f "../vkfft_test.py" ]; then
    echo "Running VKFFT Test..."
    python3 ../vkfft_test.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS
else
    echo "Skipping VKFFT Test - ../vkfft_test.py not found"
fi