#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include "../common/common.cuh"
#include "../common/nonstrided_kernels.cuh"
#include <cufft.h>

float get_bandwith_scale_factor(long long elem_count, long long fft_size) {
    return 2.0f;
}

template<int FFTSize, int FFTsInBlock, int exec_mode>
void* init_test(long long data_size, cudaStream_t stream) {
    if constexpr (exec_mode == EXEC_MODE_CUFFT) {
        const long long dim1 = FFTSize;
        const long long dim0 = data_size / dim1;

        cufftHandle* plan = new cufftHandle();

        checkCuFFT(cufftPlan1d(plan, dim1, CUFFT_C2C, dim0), "plan");
        checkCuFFT(cufftSetStream(*plan, stream), "cufftSetStream");

        return static_cast<void*>(plan);
    }

    auto config = new NonStridedFFTConfig<FFTSize, FFTsInBlock, 1>();
    config->init(stream);
    return static_cast<void*>(config);
}

template<int FFTSize, int FFTsInBlock, int exec_mode, bool validate>
void run_test(void* plan, cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems, cudaStream_t stream){
    if constexpr (exec_mode == EXEC_MODE_CUFFT) {
        checkCuFFT(cufftExecC2C(*static_cast<cufftHandle*>(plan), d_data, d_data, CUFFT_FORWARD), "exec");
        return;
    }
    
    static_cast<NonStridedFFTConfig<FFTSize, FFTsInBlock, 1>*>(plan)->execute_fft(d_data, total_elems, stream);
}

template<int FFTSize, int FFTsInBlock, int exec_mode>
void delete_test(void* plan) {
    if constexpr (exec_mode == EXEC_MODE_CUFFT) {
        cufftDestroy(*static_cast<cufftHandle*>(plan));
        delete static_cast<cufftHandle*>(plan);
        return;
    }

    delete static_cast<NonStridedFFTConfig<FFTSize, FFTsInBlock, 1>*>(plan);
}