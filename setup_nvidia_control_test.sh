#! /bin/bash

mkdir -p nvidia_control_test
cd nvidia_control_test

wget https://developer.nvidia.com/downloads/compute/cuFFTDx/redist/cuFFTDx/cuda12/nvidia-mathdx-25.06.1-cuda12.tar.gz
tar -xvf nvidia-mathdx-25.06.1-cuda12.tar.gz

git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
git checkout e6e2cc29f5e7611dfc6af0ed6409209df0068cf2
cd ..

git clone https://github.com/NVIDIA/CUDALibrarySamples.git
cd CUDALibrarySamples
git checkout a94482ebecf8b16d5b83ab276b7db3a84979f0e5

