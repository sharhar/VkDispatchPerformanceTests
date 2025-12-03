#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include "../common/common.cuh"
#include "../common/nonstrided_io.cuh"
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

float get_bandwith_scale_factor() {
    return 2.0f;
}

void make_cufft_handle(cufftHandle* plan, long long data_size, int fft_size, cudaStream_t stream) {
    
}

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void fft_kernel(cufftComplex* data, typename FFT::workspace_type workspace) {
    cufftComplex thread_data[FFT::storage_size];
    const unsigned int local_fft_id = threadIdx.y;

    load_nonstrided<FFT>(data, thread_data, local_fft_id);

    extern __shared__ __align__(alignof(float4)) cufftComplex shared_mem[];
    FFT().execute(thread_data, shared_mem, workspace);

    store_nonstrided<FFT>(thread_data, data, local_fft_id);
}

void exec_cufft_batch(cufftHandle plan, cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems, cudaStream_t stream) {
    using namespace cufftdx;

    using FFT = decltype(Block() + Size<FFT_SIZE>() + Type<fft_type::c2c>() +
                    Direction<fft_direction::forward>() +
                    Precision<float>() +
                    FFTsPerBlock<FFTS_PER_BLOCK>() + SM<ARCH>());

    cudaError_t error_code = cudaSuccess;
    auto        workspace  = make_workspace<FFT>(error_code, stream);
    CUDA_CHECK_AND_EXIT(error_code);

    // Increase shared memory size, if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        fft_kernel<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));

    fft_kernel<FFT><<<total_elems / (FFT_SIZE * FFTS_PER_BLOCK), FFT::block_dim, FFT::shared_memory_size, stream>>>(
        d_data, workspace
    );

    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
}