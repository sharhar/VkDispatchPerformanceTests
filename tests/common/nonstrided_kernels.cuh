#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include <cufft.h>

template<class FFT>
static inline __device__ void load_nonstrided(float2* data, float2* thread_data) {
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
}

template<class FFT>
static inline __device__ void store_nonstrided(float2* data, float2* thread_data) {
    const unsigned int local_fft_id = threadIdx.y;
    unsigned int global_fft_id = blockIdx.x * FFT::ffts_per_block + local_fft_id;
    const unsigned int output_offset = FFT::output_length * global_fft_id;

    #pragma unroll
    for (int i = 0; i < FFT::output_ept; ++i) {
        unsigned int element_index = i * FFT::stride + threadIdx.x;

        if(element_index >= FFT::output_length)
            break;

        data[element_index + output_offset] = thread_data[i];
    }
}

template<class FFT, int padding_ratio>
static inline __device__
void load_nonstrided_padded(const float2* input, float2* thread_data) {
    const unsigned int local_fft_id = threadIdx.y;
    unsigned int active_layers = FFT::input_length / padding_ratio;
    unsigned int extra_layers = FFT::input_length - active_layers;

    unsigned int global_fft_id = blockIdx.x * (FFT::ffts_per_block / FFT::implicit_type_batching) + local_fft_id;
    global_fft_id = global_fft_id + extra_layers * (global_fft_id / active_layers); // skip extra layers
    const unsigned int offset = FFT::input_length * global_fft_id;
    
    const unsigned int stride = FFT::stride;
    unsigned int       index  = offset + threadIdx.x;
    for (unsigned int i = 0; i < FFT::input_ept; ++i) {
        unsigned int fft_index = i * stride + threadIdx.x;

        if (fft_index < FFT::input_length / padding_ratio) {
            thread_data[i] = input[index];
            index += stride;
        } else if (fft_index < FFT::input_length) {
            thread_data[i] = float2{0.0f, 0.0f};
            index += stride;
        }
    }
}

template<class FFT, int padding_ratio>
static inline __device__
void store_nonstrided_padded(const float2* thread_data, float2* output) {
    const unsigned int local_fft_id = threadIdx.y;

    unsigned int active_layers = FFT::output_length / padding_ratio;
    unsigned int extra_layers = FFT::output_length - active_layers;
    
    unsigned int global_fft_id = blockIdx.x * (FFT::ffts_per_block / FFT::implicit_type_batching) + local_fft_id;
    global_fft_id = global_fft_id + extra_layers * (global_fft_id / active_layers); // skip extra layers

    const unsigned int offset = FFT::output_length * global_fft_id;
    const unsigned int stride = FFT::stride;
    unsigned int       index  = offset + threadIdx.x;

    for (int i = 0; i < FFT::output_ept; ++i) {
        if ((i * stride + threadIdx.x) < FFT::output_length) {
            output[index] = thread_data[i];
            index += stride;
        }
    }
}

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void nonstrided_fft_kernel(cufftComplex* data, typename FFT::workspace_type workspace) {
    cufftComplex thread_data[FFT::storage_size];
    extern __shared__ __align__(alignof(float4)) cufftComplex shared_mem[];

    load_nonstrided<FFT>(data, thread_data);
    FFT().execute(thread_data, shared_mem, workspace);
    store_nonstrided<FFT>(data, thread_data);
}

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void nonstrided_padded_fft_kernel(cufftComplex* data, typename FFT::workspace_type workspace) {
    cufftComplex thread_data[FFT::storage_size];
    extern __shared__ __align__(alignof(float4)) cufftComplex shared_mem[];

    load_nonstrided_padded<FFT, 8>(data, thread_data);
    FFT().execute(thread_data, shared_mem, workspace);
    store_nonstrided_padded<FFT, 8>(data, thread_data);
}

static inline __device__ void scaling_kernel(cufftComplex* data, float scale_factor, long long total_elems) {
    long long i = blockIdx.x * 1LL * blockDim.x + threadIdx.x;
    if (i < total_elems) {
        data[i].x *= scale_factor;
        data[i].y *= scale_factor;
    }
}

template <class FFT, class IFFT>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void nonstrided_scaled_convolution_kernel(cufftComplex* data, typename FFT::workspace_type workspace, typename IFFT::workspace_type iworkspace, float scale_factor) {
    cufftComplex thread_data[FFT::storage_size];
    extern __shared__ __align__(alignof(float4)) cufftComplex shared_mem[];

    load_nonstrided<FFT>(data, thread_data);
    FFT().execute(thread_data, shared_mem, workspace);
    
    for(int i = 0; i < FFT::storage_size; i++) {
        thread_data[i].x *= scale_factor;
        thread_data[i].y *= scale_factor;

    }

    IFFT().execute(thread_data, shared_mem, iworkspace);
    store_nonstrided<FFT>(data, thread_data);
}

template<int FFTSize, int FFTsInBlock>
struct NonStridedFFTConfig {
private:
    template<bool inverse>
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
    using FFT = decltype(make_desc<false>());
    using IFFT = decltype(make_desc<true>());
    typename FFT::workspace_type workspace;
    typename IFFT::workspace_type iworkspace;
    unsigned int shared_mem_size;
    dim3 block_dim;
    unsigned int ffts_per_block;
    
    void init(cudaStream_t stream) {
        cudaError_t err;
        workspace = cufftdx::make_workspace<FFT>(err, stream);
        CUDA_CHECK_AND_EXIT(err);

        iworkspace = cufftdx::make_workspace<IFFT>(err, stream);
        CUDA_CHECK_AND_EXIT(err);

        shared_mem_size = FFT::shared_memory_size;
        ffts_per_block = FFT::ffts_per_block;
        block_dim = FFT::block_dim;

        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            nonstrided_fft_kernel<FFT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));

        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            nonstrided_fft_kernel<IFFT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));

        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            nonstrided_padded_fft_kernel<FFT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));

        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            nonstrided_scaled_convolution_kernel<FFT, IFFT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
    }

    void execute_fft(cufftComplex* d_data, long long total_elems, cudaStream_t stream) {
        long long fft_count = total_elems / (FFTSize * FFTsInBlock);
        
        nonstrided_fft_kernel<FFT><<<fft_count, block_dim, shared_mem_size, stream>>>(
            d_data, workspace
        );

        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    }

    void execute_ifft(cufftComplex* d_data, long long total_elems, cudaStream_t stream) {
        long long fft_count = total_elems / (FFTSize * FFTsInBlock);
        
        nonstrided_fft_kernel<IFFT><<<fft_count, block_dim, shared_mem_size, stream>>>(
            d_data, workspace
        );

        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    }

    void execute_padded_fft(cufftComplex* d_data, long long total_elems, cudaStream_t stream) {
        long long fft_count = total_elems / (FFTSize * FFTsInBlock * 8);
        
        nonstrided_padded_fft_kernel<FFT><<<fft_count, block_dim, shared_mem_size, stream>>>(
            d_data, workspace
        );

        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    }

    void execute_conv(cufftComplex* d_data, long long total_elems, cudaStream_t stream, float scale_factor) {
        long long fft_count = total_elems / (FFTSize * FFTsInBlock);
        
        nonstrided_scaled_convolution_kernel<FFT, IFFT><<<fft_count, block_dim, shared_mem_size, stream>>>(
            d_data, workspace, iworkspace, scale_factor
        );

        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    }
};