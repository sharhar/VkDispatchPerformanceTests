#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include <cufft.h>

#define SMEM_BITS_PADDING 5

template<class FFT>
static inline __device__ void load_strided_smem(const float2* input,
                                            float2*          thread_data,
                                            float2*       shared_memory,
                                            unsigned int stride_len) {
    const unsigned int tid          = threadIdx.x + blockDim.x * threadIdx.y;
    const unsigned int tidx         = tid / blockDim.y;
    const unsigned int tidy         = tid % blockDim.y;

    const unsigned int batch_offset = tidy + blockIdx.y * FFT::ffts_per_block + blockIdx.x * stride_len * cufftdx::size_of<FFT>::value;

    const unsigned int stride       = stride_len * FFT::stride;
    unsigned int       index        = batch_offset + (tidx * stride_len);
    unsigned int       smem_index   = tidx + tidy * blockDim.x;
    unsigned int padded_smem_index = 0;

    #pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * FFT::stride + tidx) < cufftdx::size_of<FFT>::value) {
            padded_smem_index = smem_index + (smem_index >> SMEM_BITS_PADDING);
            shared_memory[padded_smem_index] = input[index];
            
            index += stride;
            smem_index += (blockDim.x * blockDim.y);
        }
    }
    __syncthreads();
    smem_index = threadIdx.x + threadIdx.y * blockDim.x;

    #pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * FFT::stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
            padded_smem_index = smem_index + (smem_index >> SMEM_BITS_PADDING);
            thread_data[i] = shared_memory[padded_smem_index];
            smem_index += (blockDim.x * blockDim.y);
        }
    }
}

template<class FFT>
static inline __device__ void store_strided_smem(const float2* thread_data,
                                            float2*    shared_memory,
                                            float2*    output,
                                            unsigned int stride_len) {
    unsigned int smem_index = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int padded_smem_index = 0;

    __syncthreads();
    
    #pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * FFT::stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
            padded_smem_index = smem_index + (smem_index >> SMEM_BITS_PADDING);
            shared_memory[padded_smem_index] = thread_data[i];
            smem_index += (blockDim.x * blockDim.y);
        }
    }
    __syncthreads();

    const unsigned int tid          = threadIdx.x + blockDim.x * threadIdx.y;
    const unsigned int tidx         = tid / blockDim.y;
    const unsigned int tidy         = tid % blockDim.y;
    const unsigned int batch_offset = tidy + blockIdx.y * FFT::ffts_per_block + blockIdx.x * stride_len * cufftdx::size_of<FFT>::value;
    const unsigned int stride       = stride_len * FFT::stride;
    unsigned int       index        = batch_offset + (tidx * stride_len);
    smem_index                      = tidx + tidy * blockDim.x;

    #pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * FFT::stride + tidx) < cufftdx::size_of<FFT>::value) {
            padded_smem_index = smem_index + (smem_index >> SMEM_BITS_PADDING);
            output[index] = shared_memory[padded_smem_index];

            index += stride;
            smem_index += (blockDim.x * blockDim.y);
        }
    }
}

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void strided_fft_kernel(cufftComplex* data, unsigned int inner_fft_count, typename FFT::workspace_type workspace) {
    cufftComplex thread_data[FFT::storage_size];
    extern __shared__ __align__(alignof(float4)) cufftComplex shared_mem[];
    unsigned int stride_len = inner_fft_count * FFT::ffts_per_block;

    load_strided_smem<FFT>(data, thread_data, shared_mem, stride_len);
    
    FFT().execute(thread_data, shared_mem, workspace);

    store_strided_smem<FFT>(thread_data, shared_mem, data, stride_len);
}

template<int FFTSize, int FFTsInBlock, bool inverse>
struct StridedFFTConfig {

private:
    static constexpr auto make_desc() {
        using namespace cufftdx;

        constexpr int true_ffts_per_block = FFTsInBlock >= FFTSize ? FFTSize : FFTsInBlock;

        constexpr fft_direction dir = inverse ? fft_direction::inverse : fft_direction::forward;
        auto base_desc = Block() + Size<FFTSize>() + Type<fft_type::c2c>() +
               Direction<dir>() +
               Precision<float>() +
               FFTsPerBlock<true_ffts_per_block>() + SM<ARCH>();

        using BaseFFT = decltype(base_desc);
        constexpr int default_ept = BaseFFT::elements_per_thread;

        constexpr int target_ept = default_ept; // (FFTSize < 64) ? 4 : default_ept;

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
        
        unsigned int user_staging_size = FFT::block_dim.x * FFT::block_dim.y * FFT::elements_per_thread * sizeof(cufftComplex);
        user_staging_size = user_staging_size + (user_staging_size >> SMEM_BITS_PADDING); // padding for avoiding bank conflicts

        shared_mem_size = std::max(user_staging_size, (unsigned int)FFT::shared_memory_size);
        ffts_per_block = FFT::ffts_per_block;
        block_dim = FFT::block_dim;

        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            strided_fft_kernel<FFT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
    }

    void execute(cufftComplex* d_data, long long total_elems, cudaStream_t stream) {
        long long outer_fft_count = total_elems / (FFTSize * FFTSize);
        long long inner_fft_count = FFTSize / ffts_per_block;
        
        dim3 grid_dims(outer_fft_count, inner_fft_count);
        
        strided_fft_kernel<FFT><<<grid_dims, block_dim, shared_mem_size, stream>>>(
            d_data, inner_fft_count, workspace
        );

        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    }
};