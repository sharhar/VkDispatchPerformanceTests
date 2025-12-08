#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include <cufft.h>

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void nonstrided_fft_kernel(cufftComplex* data, typename FFT::workspace_type workspace) {
    cufftComplex thread_data[FFT::storage_size];
    extern __shared__ __align__(alignof(float4)) cufftComplex shared_mem[];
    const unsigned int local_fft_id = threadIdx.y;

    unsigned int global_fft_id = blockIdx.x * FFT::ffts_per_block + local_fft_id;
    const unsigned int input_offset = FFT::input_length * global_fft_id;

    #pragma unroll
    for (unsigned int i = 0; i < FFT::input_ept; i++) {
        unsigned int element_index = i * FFT::stride + threadIdx.x;

        if(element_index >= FFT::input_length)
            break;

        thread_data[i] = data[element_index + input_offset];
    }
    
    FFT().execute(thread_data, shared_mem, workspace);

    const unsigned int output_offset = FFT::output_length * global_fft_id;

    #pragma unroll
    for (int i = 0; i < FFT::output_ept; ++i) {
        unsigned int element_index = i * FFT::stride + threadIdx.x;

        if(element_index >= FFT::output_length)
            break;

        data[element_index + output_offset] = thread_data[i];
    }
}

template<int FFTSize, int FFTsInBlock, bool inverse>
struct NonStridedFFTConfig {
private:
    static constexpr auto make_desc() {
        using namespace cufftdx;

        constexpr fft_direction dir = inverse ? fft_direction::inverse : fft_direction::forward;
        auto base_desc = Block() + Size<FFTSize>() + Type<fft_type::c2c>() +
               Direction<dir>() +
               Precision<float>() +
               FFTsPerBlock<FFTsInBlock>() + SM<ARCH>();

        using BaseFFT = decltype(base_desc);
        constexpr int default_ept = BaseFFT::elements_per_thread;

        constexpr int target_ept = (FFTSize < 64) ? 4 : default_ept;

        return base_desc + ElementsPerThread<target_ept>();
    }
public:
    using FFT = decltype(make_desc());
    typename FFT::workspace_type workspace;
    unsigned int shared_mem_size;
    dim3 block_dim;
    unsigned int ffts_per_block;
    
    void init(cudaStream_t stream) {
        cudaError_t err;
        workspace = cufftdx::make_workspace<FFT>(err, stream);

        shared_mem_size = FFT::shared_memory_size;
        ffts_per_block = FFT::ffts_per_block;
        block_dim = FFT::block_dim;

        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            nonstrided_fft_kernel<FFT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
    }

    void execute(cufftComplex* d_data, long long total_elems, cudaStream_t stream) {
        long long fft_count = total_elems / (FFTSize * FFTsInBlock);
        
        nonstrided_fft_kernel<FFT><<<fft_count, block_dim, shared_mem_size, stream>>>(
            d_data, workspace
        );

        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    }
};