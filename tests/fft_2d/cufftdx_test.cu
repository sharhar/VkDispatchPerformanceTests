#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include "../common/common.cuh"
#include "../common/strided_kernels.cuh"
#include "../common/nonstrided_kernels.cuh"
#include <cufft.h>

float get_bandwith_scale_factor(long long elem_count, long long fft_size) {
    return 4.0f;
}

template<int FFTSize, int FFTsInBlock>
struct FFT2DConfig {
    NonStridedFFTConfig<FFTSize, FFTsInBlock, 1> fft_nonstrided;
    StridedFFTConfig<FFTSize, FFTsInBlock, true, false, 1> fft_strided;
};

template<int FFTSize, int FFTsInBlock, int exec_mode>
void* init_test(long long data_size, cudaStream_t stream) {
    if constexpr (exec_mode == EXEC_MODE_CUFFT) {
        cufftHandle* plan = new cufftHandle();

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

        checkCuFFT(cufftPlanMany(plan, 2, n,
                                    inembed,  istride, idist,
                                    onembed,  ostride, odist,
                                    CUFFT_C2C, int(dim0)), "plan2d");
        checkCuFFT(cufftSetStream(*plan, stream), "cufftSetStream");

        return static_cast<void*>(plan);
    }

    auto config = new FFT2DConfig<FFTSize, FFTsInBlock>();
    
    config->fft_nonstrided.init(stream);
    config->fft_strided.init(stream);
    
    return static_cast<void*>(config);
}

template<int FFTSize, int FFTsInBlock, int exec_mode, bool validate>
void run_test(void* plan, cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems, cudaStream_t stream){
    if constexpr (exec_mode == EXEC_MODE_CUFFT) {
        checkCuFFT(cufftExecC2C(*static_cast<cufftHandle*>(plan), d_data, d_data, CUFFT_FORWARD), "exec");
        return;
    }
    
    static_cast<FFT2DConfig<FFTSize, FFTsInBlock>*>(plan)->fft_nonstrided.execute_fft(d_data, total_elems, stream);
    static_cast<FFT2DConfig<FFTSize, FFTsInBlock>*>(plan)->fft_strided.execute_fft(d_data, total_elems, stream);
}

template<int FFTSize, int FFTsInBlock, int exec_mode>
void delete_test(void* plan) {
    if constexpr (exec_mode == EXEC_MODE_CUFFT) {
        cufftDestroy(*static_cast<cufftHandle*>(plan));
        delete static_cast<cufftHandle*>(plan);
        return;
    }

    delete static_cast<FFT2DConfig<FFTSize, FFTsInBlock>*>(plan);
}