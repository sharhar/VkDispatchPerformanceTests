#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include "../common/common.cuh"
#include "../common/strided_kernels.cuh"
#include "../common/nonstrided_kernels.cuh"
#include <cufft.h>

float get_bandwith_scale_factor() {
    return 11.0f;
}

template<int FFTSize, int FFTsInBlock>
struct FFTConv2DConfig {
    NonStridedFFTConfig<FFTSize, FFTsInBlock, 1> fft_nonstrided;
    StridedFFTConfig<FFTSize, FFTsInBlock, true, false, 1> fft_strided;
};

template<int FFTSize, int FFTsInBlock, bool reference_mode>
void* init_test(long long data_size, cudaStream_t stream) {
    if constexpr (reference_mode) {
        const long long dim2 = FFTSize;
        const long long dim1 = FFTSize;
        const long long dim0 = data_size / (dim1 * dim2);

        int n[2] = { int(dim1), int(dim2) };
        int inembed[2] = { int(dim1), int(dim2) };
        int onembed[2] = { int(dim1), int(dim2) };
        int istride    = 1;
        int ostride    = 1;
        int idist      = int(dim1)* int(dim2);
        int odist      = int(dim1)* int(dim2);

        cufftHandle* plan = new cufftHandle();

        checkCuFFT(cufftPlanMany(plan, 2, n,
                                    inembed,  istride, idist,
                                    onembed,  ostride, odist,
                                    CUFFT_C2C, int(dim0)), "plan2d");

        checkCuFFT(cufftSetStream(*plan, stream), "cufftSetStream");

        return plan;
    }

    auto config = new FFTConv2DConfig<FFTSize, FFTsInBlock>();
    
    config->fft_nonstrided.init(stream);
    config->fft_strided.init(stream);
    
    return static_cast<void*>(config);
}


__global__ void convolve_arrays(cufftComplex* data, cufftComplex* kernel, long long total_elems) {
    long long i = blockIdx.x * 1LL * blockDim.x + threadIdx.x;
    if (i < total_elems) {
        const size_t idx_in_image = i;
        const cufftComplex d = data[i];
        const cufftComplex k = kernel[idx_in_image];

        const float real = d.x * k.x - d.y * k.y;
        const float imag = d.x * k.y + d.y * k.x;
        data[i] = make_float2(real, imag);
    }
}

template<int FFTSize, int FFTsInBlock, bool reference_mode, bool validate>
void run_test(void* plan, cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems, cudaStream_t stream){
    if constexpr (reference_mode) {
        checkCuFFT(cufftExecC2C(*static_cast<cufftHandle*>(plan), d_data, d_data, CUFFT_FORWARD), "exec");
        convolve_arrays<<<(total_elems+255)/256,256,0,stream>>>(d_data, d_kernel, total_elems);
        checkCuFFT(cufftExecC2C(*static_cast<cufftHandle*>(plan), d_data, d_data, CUFFT_INVERSE), "exec");
        return;
    }
    
    static_cast<FFTConv2DConfig<FFTSize, FFTsInBlock>*>(plan)->fft_nonstrided.execute_fft(d_data, total_elems, stream);
    static_cast<FFTConv2DConfig<FFTSize, FFTsInBlock>*>(plan)->fft_strided.execute_conv(d_data, d_kernel, total_elems, stream);
    static_cast<FFTConv2DConfig<FFTSize, FFTsInBlock>*>(plan)->fft_nonstrided.execute_ifft(d_data, total_elems, stream);
}


template<int FFTSize, int FFTsInBlock, bool reference_mode>
void delete_test(void* plan) {
    if constexpr (reference_mode) {
        cufftDestroy(*static_cast<cufftHandle*>(plan));
        delete static_cast<cufftHandle*>(plan);
        return;
    }

    delete static_cast<FFTConv2DConfig<FFTSize, FFTsInBlock>*>(plan);
}