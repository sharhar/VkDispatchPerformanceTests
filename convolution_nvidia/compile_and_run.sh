#! /bin/bash

# ensure $3 is a positive integer
case "$3" in
  ''|*[!0-9]*) echo "Second arg must be a positive integer"; exit 64;;
  0) echo "Repeat count must be >= 1"; exit 64;;
esac

if [ -n "$CUDA_HOME" ]; then
    NVCC="$CUDA_HOME/bin/nvcc"
else
    NVCC="nvcc"
fi

fft_size="$1"

ARCH="$2"

file="../convolution_nvidia_dependencies/CUDALibrarySamples/MathDx/cuFFTDx/06_convolution/convolution_performance.cu"
line_no=228

if [ ! -f "$file" ]; then
  echo "Error: file '$file' not found" >&2
  exit 2
fi

# use awk to replace exactly line $line_no
tmp="$(mktemp "${file}.tmp.XXXXXX")"
awk -v N="$line_no" -v VAL="$fft_size" \
    'NR==N { printf("static constexpr unsigned int fft_size = %s;\n", VAL); next } { print }' \
    "$file" > "$tmp"

# move into place
mv -- "$tmp" "$file"
echo "Replaced line $line_no in $file"

$NVCC -std=c++17 -O3 \
    -I ../convolution_nvidia_dependencies/cutlass/include \
    -I ../convolution_nvidia_dependencies/nvidia-mathdx-25.06.1/nvidia/mathdx/25.06/include \
    -DCUFFTDX_EXAMPLE_ENABLE_SM_${ARCH} \
    -DCUFFTDX_EXAMPLE_CMAKE \
    -gencode arch=compute_${ARCH},code=sm_${ARCH} \
    -lcufft \
    $file -o test_results/nvidia_convolution_test.exec


repeat="$3"

for ((i=1; i<=repeat; i++)); do
  echo ""
  echo ""
  echo "Iteration $i | fft_size=$1"
  echo "============================="
  ./test_results/nvidia_convolution_test.exec
done

rm test_results/nvidia_convolution_test.exec