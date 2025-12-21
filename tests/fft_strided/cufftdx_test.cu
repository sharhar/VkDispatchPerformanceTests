#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include "../common/common.cuh"
#include "../common/strided_kernels.cuh"
#include <cufft.h>

float get_bandwith_scale_factor(long long elem_count, long long fft_size) {
    return 2.0f;
}

template<int FFTSize, int FFTsInBlock, bool reference_mode>
void* init_test(long long data_size, cudaStream_t stream) {
    auto config = new StridedFFTConfig<FFTSize, FFTsInBlock, true, false, 1>();
    config->init(stream);
    return static_cast<void*>(config);
}

template<int FFTSize, int FFTsInBlock, bool reference_mode, bool validate>
void run_test(void* plan, cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems, cudaStream_t stream){
    static_cast<StridedFFTConfig<FFTSize, FFTsInBlock, true, false, 1>*>(plan)->execute_fft(d_data, total_elems, stream);
}

template<int FFTSize, int FFTsInBlock, bool reference_mode>
void delete_test(void* plan) {
    delete static_cast<StridedFFTConfig<FFTSize, FFTsInBlock, true, false, 1>*>(plan);
}