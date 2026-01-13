#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include "../common/common.cuh"
#include "../common/nonstrided_kernels.cuh"
#include <cufft.h>

float get_bandwith_scale_factor(long long elem_count, long long fft_size) {
    return 6.0f;
}

template<int FFTSize, int FFTsInBlock, int exec_mode>
void* init_test(long long data_size, cudaStream_t stream) {
    if constexpr (exec_mode == EXEC_MODE_CUFFT) {
        cufftHandle* plan = new cufftHandle();

        const long long dim1 = FFTSize;
        const long long dim0 = data_size / dim1;

        checkCuFFT(cufftPlan1d(plan, dim1, CUFFT_C2C, dim0), "plan");
        checkCuFFT(cufftSetStream(*plan, stream), "cufftSetStream");

        return static_cast<void*>(plan);
    }

    auto config = new NonStridedFFTConfig<FFTSize, FFTsInBlock, 1>();
    config->init(stream);
    return static_cast<void*>(config);
}

__global__ void scale_kernel(cufftComplex* data, float scale_factor, long long total_elems) {
    long long i = blockIdx.x * 1LL * blockDim.x + threadIdx.x;
    if (i < total_elems) {
        data[i].x *= scale_factor;
        data[i].y *= scale_factor;
    }
}

template<int FFTSize, int FFTsInBlock, int exec_mode, bool validate>
void run_test(void* plan, cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems, cudaStream_t stream){
    if constexpr (exec_mode == EXEC_MODE_CUFFT) {
        checkCuFFT(cufftExecC2C(*static_cast<cufftHandle*>(plan), d_data, d_data, CUFFT_FORWARD), "exec");
        scale_kernel<<<(total_elems+255)/256,256,0,stream>>>(d_data, 0.1f, total_elems);
        checkCuFFT(cufftExecC2C(*static_cast<cufftHandle*>(plan), d_data, d_data, CUFFT_INVERSE), "exec");
        return;
    }

    if constexpr (exec_mode == EXEC_MODE_CUFFTDX_NAIVE) {
        static_cast<NonStridedFFTConfig<FFTSize, FFTsInBlock, 1>*>(plan)->execute_fft(d_data, total_elems, stream);
        scale_kernel<<<(total_elems+255)/256,256,0,stream>>>(d_data, 0.1f, total_elems);
        static_cast<NonStridedFFTConfig<FFTSize, FFTsInBlock, 1>*>(plan)->execute_ifft(d_data, total_elems, stream);
        return;
    }
    
    static_cast<NonStridedFFTConfig<FFTSize, FFTsInBlock, 1>*>(plan)->execute_conv(d_data, total_elems, stream, 0.1f);
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