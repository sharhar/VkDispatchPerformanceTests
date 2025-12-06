#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include "../common/common.cuh"
#include "../common/strided_fft.cuh"
#include <cufft.h>

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
    strided_fft<FFTSize, FFTsInBlock, false>(plan, d_data, total_elems / (FFTSize * FFTSize), FFTSize / FFTsInBlock, stream);
}