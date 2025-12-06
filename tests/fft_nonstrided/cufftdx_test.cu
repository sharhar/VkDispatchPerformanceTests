#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include "../common/common.cuh"
#include "../common/nonstrided_fft.cuh"
#include <cufft.h>

#ifndef FFT_SIZE
#define FFT_SIZE 1024
#endif

#ifndef FFTS_PER_BLOCK
#define FFTS_PER_BLOCK 4
#endif

#ifndef ARCH
#define ARCH 800 // Example: sm_80
#endif

const char* get_test_name() {
    return "cufftdx";
}

float get_bandwith_scale_factor() {
    return 2.0f;
}

template<int FFTSize>
void make_cufft_handle(cufftHandle* plan, long long data_size, cudaStream_t stream) {
    
}

template<int FFTSize, int FFTsInBlock>
void exec_cufft_batch(cufftHandle plan, cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems, cudaStream_t stream) {
    nonstrided_fft<FFTSize, FFTsInBlock, false>(plan, d_data, total_elems / (FFTSize * FFTsInBlock), stream);
}