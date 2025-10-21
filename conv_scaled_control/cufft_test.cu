#include "../common/common.cuh"

float get_bandwith_scale_factor() {
    return 6.0f;
}

void make_cufft_handle(cufftHandle* plan, long long data_size, int fft_size) {
    const long long dim1 = fft_size;
    const long long dim0 = data_size / dim1;

    //checkCuFFT(cufftCreate(plan), "cufftCreate");
    checkCuFFT(cufftPlan1d(plan, dim1, CUFFT_C2C, dim0), "plan");
}

__global__ void scale_kernel(cufftComplex* data, float scale_factor, long long total_elems) {
    long long i = blockIdx.x * 1LL * blockDim.x + threadIdx.x;
    if (i < total_elems) {
        data[i].x *= scale_factor;
        data[i].y *= scale_factor;
    }
}

void exec_cufft_batch(cufftHandle plan, cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems) {
    checkCuFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD), "warmup");
    scale_kernel<<<(total_elems+255)/256,256>>>(d_data, 5.0, total_elems);
    checkCuFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE), "warmup");
}