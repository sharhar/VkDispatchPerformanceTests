#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include "../common/common.cuh"
#include "../common/nonstrided_fft.cuh"
#include <cufft.h>

const char* get_test_name() {
    return "cufftdx";
}

float get_bandwith_scale_factor() {
    return 2.0f;
}

template<int FFTSize, int FFTsInBlock>
void* init_test(long long data_size, cudaStream_t stream) {
    auto config = new NonStridedFFTConfig<FFTSize, FFTsInBlock, false>();
    config->init(stream);
    return static_cast<void*>(config);
}

template<int FFTSize, int FFTsInBlock>
void run_test(void* plan, cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems, cudaStream_t stream){
    static_cast<NonStridedFFTConfig<FFTSize, FFTsInBlock, false>*>(plan)->execute(d_data, total_elems, stream);
}