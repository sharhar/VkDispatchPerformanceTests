#include "../common/common.cuh"

float get_bandwith_scale_factor() {
    return 2.0f;
}

void make_cufft_handle(cufftHandle* plan, long long data_size, int fft_size) {
    //const long long dim1 = fft_size;
    //const long long dim0 = data_size / dim1;

    //checkCuFFT(cufftCreate(plan), "cufftCreate");
    //checkCuFFT(cufftPlan1d(plan, dim1, CUFFT_C2C, dim0), "plan");
}

__global__ void add_kernel(float2* data, unsigned int n) {
    unsigned int tid    = blockIdx.x * blockDim.x + threadIdx.x;

    float2 v = data[tid];
    v.x += 1.0f;
    v.y += 1.0f;
    data[tid] = v;
}

void exec_cufft_batch(cufftHandle plan, cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems) {
    add_kernel<<<(total_elems + 1023) / 1024, 1024>>>(d_data, total_elems);
}