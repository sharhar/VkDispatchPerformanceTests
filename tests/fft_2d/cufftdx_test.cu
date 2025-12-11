#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include "../common/common.cuh"
#include "../common/strided_kernels.cuh"
#include "../common/nonstrided_kernels.cuh"
#include <cufft.h>

const char* get_test_name() {
    return "cufftdx";
}

float get_bandwith_scale_factor() {
    return 4.0f;
}

template<int FFTSize, int FFTsInBlock>
struct FFT2DConfig {
    NonStridedFFTConfig<FFTSize, FFTsInBlock> fft_nonstrided;
    StridedFFTConfig<FFTSize, FFTsInBlock, true, false> fft_strided;
};

template<int FFTSize, int FFTsInBlock>
void* init_test(long long data_size, cudaStream_t stream) {
    auto config = new FFT2DConfig<FFTSize, FFTsInBlock>();
    
    config->fft_nonstrided.init(stream);
    config->fft_strided.init(stream);
    
    return static_cast<void*>(config);
}

template<int FFTSize, int FFTsInBlock>
void run_test(void* plan, cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems, cudaStream_t stream){
    static_cast<FFT2DConfig<FFTSize, FFTsInBlock>*>(plan)->fft_nonstrided.execute_fft(d_data, total_elems, stream);
    static_cast<FFT2DConfig<FFTSize, FFTsInBlock>*>(plan)->fft_strided.execute_fft(d_data, total_elems, stream);
}