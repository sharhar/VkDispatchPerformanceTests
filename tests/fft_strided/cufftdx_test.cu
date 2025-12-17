#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include "../common/common.cuh"
#include "../common/strided_kernels.cuh"
#include <cufft.h>

float get_bandwith_scale_factor() {
    return 2.0f;
}

template<int FFTSize, int FFTsInBlock, bool reference_mode>
void* init_test(long long data_size, cudaStream_t stream) {
    auto config = new StridedFFTConfig<FFTSize, FFTsInBlock, true, false>();
    config->init(stream);
    return static_cast<void*>(config);
}

template<int FFTSize, int FFTsInBlock, bool reference_mode>
void run_test(void* plan, cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems, cudaStream_t stream){
    static_cast<StridedFFTConfig<FFTSize, FFTsInBlock, true, false>*>(plan)->execute_fft(d_data, total_elems, stream);
}

template<int FFTSize, int FFTsInBlock, bool reference_mode>
void delete_test(void* plan) {
    delete static_cast<StridedFFTConfig<FFTSize, FFTsInBlock, true, false>*>(plan);
}