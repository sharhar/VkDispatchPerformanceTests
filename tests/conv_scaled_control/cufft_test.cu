#include "../common/common.cuh"

const char* get_test_name() {
    return "cufft";
}

float get_bandwith_scale_factor() {
    return 6.0f;
}

template<int FFTSize, int FFTsInBlock>
void* init_test(long long data_size, cudaStream_t stream) {
    cufftHandle* plan = new cufftHandle();

    const long long dim1 = FFTSize;
    const long long dim0 = data_size / dim1;

    checkCuFFT(cufftPlan1d(plan, dim1, CUFFT_C2C, dim0), "plan");
    checkCuFFT(cufftSetStream(*plan, stream), "cufftSetStream");

    return static_cast<void*>(plan);
}

__global__ void scale_kernel(cufftComplex* data, float scale_factor, long long total_elems) {
    long long i = blockIdx.x * 1LL * blockDim.x + threadIdx.x;
    if (i < total_elems) {
        data[i].x *= scale_factor;
        data[i].y *= scale_factor;
    }
}

template<int FFTSize, int FFTsInBlock>
void run_test(void* plan, cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems, cudaStream_t stream){
    checkCuFFT(cufftExecC2C(*static_cast<cufftHandle*>(plan), d_data, d_data, CUFFT_FORWARD), "exec");
    scale_kernel<<<(total_elems+255)/256,256,0,stream>>>(d_data, 5.0, total_elems);
    checkCuFFT(cufftExecC2C(*static_cast<cufftHandle*>(plan), d_data, d_data, CUFFT_FORWARD), "exec");
}