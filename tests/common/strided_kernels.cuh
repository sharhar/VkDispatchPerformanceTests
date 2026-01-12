#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include <cufft.h>

#define SMEM_BITS_PADDING 5

template<class FFT, bool smem_transpose>
static inline __device__ void load_strided(const float2* input,
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
            if constexpr (smem_transpose) {
                padded_smem_index = smem_index + (smem_index >> SMEM_BITS_PADDING);
                shared_memory[padded_smem_index] = input[index];
                smem_index += (blockDim.x * blockDim.y);
            } else {
                thread_data[i] = input[index];
            }
            
            index += stride;
        }
    }

    if constexpr (!smem_transpose) return;

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

template<class FFT, int padding_ratio>
static inline __device__ void load_strided_padded(const float2* input,
                                            float2*          thread_data,
                                            float2*       shared_memory,
                                            unsigned int stride_len) {
    const unsigned int tid          = threadIdx.x + blockDim.x * threadIdx.y;
    const unsigned int tidx         = tid / blockDim.y;
    const unsigned int tidy         = tid % blockDim.y;
    
    constexpr unsigned int signal_len = cufftdx::size_of<FFT>::value / padding_ratio;
    constexpr unsigned int max_iters = (signal_len + FFT::stride - 1) / FFT::stride;
    const unsigned int batch_offset = tidy + blockIdx.y * FFT::ffts_per_block + blockIdx.x * stride_len * cufftdx::size_of<FFT>::value;

    const unsigned int stride       = stride_len * FFT::stride;
    unsigned int       index        = batch_offset + (tidx * stride_len);
    unsigned int       smem_index   = tidx + tidy * blockDim.x;
    unsigned int padded_smem_index = 0;

    #pragma unroll
    for (unsigned int i = 0; i < max_iters; i++) {
        unsigned int fft_index = i * FFT::stride + tidx;

        if (fft_index < signal_len) {
            padded_smem_index = smem_index + (smem_index >> SMEM_BITS_PADDING);
            shared_memory[padded_smem_index] = input[index];

            index += stride;
            smem_index += (blockDim.x * blockDim.y);
        }
    }

    __syncthreads();

    #pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        thread_data[i] = float2{0.0f, 0.0f};
    }

    smem_index = threadIdx.x + threadIdx.y * blockDim.x;
    #pragma unroll
    for (unsigned int i = 0; i < max_iters; i++) {
        unsigned int fft_index = i * FFT::stride + threadIdx.x;

        if (fft_index < signal_len) {
            padded_smem_index = smem_index + (smem_index >> SMEM_BITS_PADDING);
            thread_data[i] = shared_memory[padded_smem_index];
            smem_index += (blockDim.x * blockDim.y);
        }
        
        //else if (fft_index < cufftdx::size_of<FFT>::value) {
        //    thread_data[i] = float2{0.0f, 0.0f};
        //    smem_index += (blockDim.x * blockDim.y);
        //}
    }
}

template<class FFT, bool smem_transpose>
static inline __device__ void store_strided(const float2* thread_data,
                                            float2*    shared_memory,
                                            float2*    output,
                                            unsigned int stride_len) {
    unsigned int smem_index = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int padded_smem_index = 0;
    
    if constexpr (smem_transpose) {
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
    }

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
            if constexpr (smem_transpose) {
                padded_smem_index = smem_index + (smem_index >> SMEM_BITS_PADDING);
                output[index] = shared_memory[padded_smem_index];
                smem_index += (blockDim.x * blockDim.y);
            } else {
                output[index] = thread_data[i];
            }
            
            index += stride;
        }
    }
}

template <class FFT, bool smem_transpose>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void strided_fft_kernel(cufftComplex* data, unsigned int inner_fft_count, typename FFT::workspace_type workspace) {
    cufftComplex thread_data[FFT::storage_size];
    extern __shared__ __align__(alignof(float4)) cufftComplex shared_mem[];
    unsigned int stride_len = inner_fft_count * FFT::ffts_per_block;

    load_strided<FFT, smem_transpose>(data, thread_data, shared_mem, stride_len);
    
    FFT().execute(thread_data, shared_mem, workspace);

    store_strided<FFT, smem_transpose>(thread_data, shared_mem, data, stride_len);
}

template<class FFT, bool smem_transpose, bool read_kernel_transposed, bool multi_layer_kernel> 
__device__ void apply_kernel(float2* kernel, float2* thread_data, float2* shared_mem, unsigned int inner_batch_count) {
    if constexpr (read_kernel_transposed) {
        const size_t kernel_stride       = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
        size_t       kernel_index        = threadIdx.x + blockDim.x * threadIdx.y;
        kernel_index += (blockIdx.x + gridDim.x * blockIdx.y) * blockDim.x * blockDim.y;

        // complex multiplication in the frequency domain
        for (unsigned int i = 0; i < FFT::elements_per_thread; ++i) {
            float2 kernel_thread_data;// = kernel[kernel_index];

            if constexpr (multi_layer_kernel) {
                kernel_thread_data = kernel[kernel_index];
            } else {
                kernel_thread_data = kernel[kernel_index % (cufftdx::size_of<FFT>::value * cufftdx::size_of<FFT>::value)];
            }

            kernel_index += kernel_stride;

            float2 a;
            a.x = thread_data[i].x;
            a.y = thread_data[i].y;

            float2 b;
            b.x = kernel_thread_data.x;
            b.y = kernel_thread_data.y;
            
            float2 c;
            c.x = a.x * b.x - a.y * b.y;
            c.y = a.x * b.y + a.y * b.x;

            thread_data[i].x = c.x;
            thread_data[i].y = c.y;
        }
    } else {
        // Local array for thread
        float2 kernel_thread_data[FFT::storage_size];

        if constexpr (smem_transpose)
            __syncthreads();

        load_strided<FFT, smem_transpose>(kernel, kernel_thread_data, shared_mem, inner_batch_count * FFT::ffts_per_block);

        if constexpr (smem_transpose)
            __syncthreads();

        // complex multiplication in the frequency domain
        for (unsigned int i = 0; i < FFT::elements_per_thread; ++i) {
            float2 a;
            a.x = thread_data[i].x;
            a.y = thread_data[i].y;

            float2 b;
            b.x = kernel_thread_data[i].x;
            b.y = kernel_thread_data[i].y;
            
            float2 c;
            c.x = a.x * b.x - a.y * b.y;
            c.y = a.x * b.y + a.y * b.x;

            thread_data[i].x = c.x;
            thread_data[i].y = c.y;
        }
    }
}

template <class FFT, class IFFT, bool smem_transpose, bool read_kernel_transposed>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void strided_conv_kernel(cufftComplex* data, const cufftComplex* kernel, unsigned int inner_fft_count, typename FFT::workspace_type workspace, typename FFT::workspace_type iworkspace) {
    cufftComplex thread_data[FFT::storage_size];
    extern __shared__ __align__(alignof(float4)) cufftComplex shared_mem[];
    unsigned int stride_len = inner_fft_count * FFT::ffts_per_block;

    load_strided<FFT, smem_transpose>(data, thread_data, shared_mem, stride_len);
    
    FFT().execute(thread_data, shared_mem, workspace);

    apply_kernel<FFT, smem_transpose, read_kernel_transposed, true>( (float2*)kernel, (float2*)thread_data, (float2*)shared_mem, inner_fft_count);

    IFFT().execute(thread_data, shared_mem, iworkspace);

    store_strided<FFT, smem_transpose>(thread_data, shared_mem, data, stride_len);
}

template <class FFT, class IFFT, bool smem_transpose, bool read_kernel_transposed, int padding_ratio>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void strided_padded_conv_kernel(cufftComplex* data, const cufftComplex* kernel, unsigned int inner_fft_count, typename FFT::workspace_type workspace, typename FFT::workspace_type iworkspace) {
    cufftComplex thread_data[FFT::storage_size];
    extern __shared__ __align__(alignof(float4)) cufftComplex shared_mem[];
    unsigned int stride_len = inner_fft_count * FFT::ffts_per_block;

    load_strided_padded<FFT, padding_ratio>(data, thread_data, shared_mem, stride_len);
    
    FFT().execute(thread_data, shared_mem, workspace);

    apply_kernel<FFT, smem_transpose, read_kernel_transposed, true>( (float2*)kernel, (float2*)thread_data, (float2*)shared_mem, inner_fft_count);

    IFFT().execute(thread_data, shared_mem, iworkspace);

    store_strided<FFT, smem_transpose>(thread_data, shared_mem, data, stride_len);
}

template<int FFTSize, int FFTsInBlock, bool smem_transpose, bool read_kernel_transposed, int padding_ratio>
struct StridedFFTConfig {

private:
    template<bool inverse>
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
    using FFT = decltype(make_desc<false>());
    using IFFT = decltype(make_desc<true>());
    typename FFT::workspace_type workspace;
    typename FFT::workspace_type iworkspace;
    unsigned int shared_mem_size;
    dim3 block_dim;
    unsigned int ffts_per_block;
    
    void init(cudaStream_t stream) {
        cudaError_t err;
        workspace = cufftdx::make_workspace<FFT>(err, stream);
        CUDA_CHECK_AND_EXIT(err);

        iworkspace = cufftdx::make_workspace<IFFT>(err, stream);
        CUDA_CHECK_AND_EXIT(err);
        
        unsigned int user_staging_size = FFT::block_dim.x * FFT::block_dim.y * FFT::elements_per_thread * sizeof(cufftComplex);
        user_staging_size = user_staging_size + (user_staging_size >> SMEM_BITS_PADDING); // padding for avoiding bank conflicts

        shared_mem_size = std::max(user_staging_size, (unsigned int)FFT::shared_memory_size);
        ffts_per_block = FFT::ffts_per_block;
        block_dim = FFT::block_dim;

        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            strided_fft_kernel<FFT, smem_transpose>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));

        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            strided_conv_kernel<FFT, IFFT, smem_transpose, read_kernel_transposed>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));

        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            strided_padded_conv_kernel<FFT, IFFT, smem_transpose, read_kernel_transposed, padding_ratio>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
    }

    void execute_fft(cufftComplex* d_data, long long total_elems, cudaStream_t stream) {
        long long outer_fft_count = total_elems / (FFTSize * FFTSize);
        long long inner_fft_count = FFTSize / ffts_per_block;
        
        dim3 grid_dims(outer_fft_count, inner_fft_count);
        
        strided_fft_kernel<FFT, smem_transpose><<<grid_dims, block_dim, shared_mem_size, stream>>>(
            d_data, inner_fft_count, workspace
        );

        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    }

    void execute_conv(cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems, cudaStream_t stream) {
        long long outer_fft_count = total_elems / (FFTSize * FFTSize);
        long long inner_fft_count = FFTSize / ffts_per_block;
        
        dim3 grid_dims(outer_fft_count, inner_fft_count);
        
        strided_conv_kernel<FFT, IFFT, smem_transpose, read_kernel_transposed><<<grid_dims, block_dim, shared_mem_size, stream>>>(
            d_data, d_kernel, inner_fft_count, workspace, iworkspace
        );

        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    }

    void execute_padded_conv(cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems, cudaStream_t stream) {
        long long outer_fft_count = total_elems / (FFTSize * FFTSize);
        long long inner_fft_count = FFTSize / ffts_per_block;
        
        dim3 grid_dims(outer_fft_count, inner_fft_count);
        
        strided_padded_conv_kernel<FFT, IFFT, smem_transpose, read_kernel_transposed, padding_ratio><<<grid_dims, block_dim, shared_mem_size, stream>>>(
            d_data, d_kernel, inner_fft_count, workspace, iworkspace
        );

        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    }
};