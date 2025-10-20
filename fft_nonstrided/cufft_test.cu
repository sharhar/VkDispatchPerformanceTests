#include "../common/common.cuh"

float get_bandwith_scale_factor() {
    return 2.0f;
}

void make_cufft_handle(cufftHandle* plan, long long data_size, int fft_size) {
    const long long dim1 = fft_size;
    const long long dim0 = data_size / dim1;

    checkCuFFT(cufftCreate(plan), "cufftCreate");
    checkCuFFT(cufftPlan1d(plan, dim1, CUFFT_C2C, dim0), "plan");
}

void exec_cufft_batch(cufftHandle plan, cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems) {
    checkCuFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD), "exec");
}