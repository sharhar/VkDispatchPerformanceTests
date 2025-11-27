#! /bin/bash

mkdir -p dependencies
cd dependencies

if [ ! -f "vulkansdk-linux-x86_64-1.4.328.1.tar.xz" ]; then
    wget https://sdk.lunarg.com/sdk/download/1.4.328.1/linux/vulkansdk-linux-x86_64-1.4.328.1.tar.xz
    tar -xvf vulkansdk-linux-x86_64-1.4.328.1.tar.xz
fi

if [ ! -d "VkFFT" ]; then
    git clone https://github.com/DTolm/VkFFT.git
    cd VkFFT
    git checkout 066a17c17068c0f11c9298d848c2976c71fad1c1
    cd ..
fi

# Do Vulkan Test
cd VkFFT
mkdir -p build
cd build
rm -rf *
VULKAN_SDK=../1.4.328.1/x86_64/ cmake ..
make -j8
./VkFFT_TestSuite -vkfft 0 > ../../../vkfft_control_vulkan_output.log
cd ..

mkdir -p build_cuda
cd build_cuda
rm -rf *
cmake .. -DVKFFT_BACKEND=1
make -j8
./VkFFT_TestSuite -vkfft 0 > ../../../vkfft_control_cuda_output.log
./VkFFT_TestSuite -cufft 0 > ../../../vkfft_control_cufft_output.log

cd ..
cd ..

cd ..

mkdir -p test_results

python3 parse_logs.py