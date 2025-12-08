#include "../common/common.cuh"

const char* get_test_name() {
    return "cufft";
}

float get_bandwith_scale_factor() {
    return 2.0f;
}

template<int FFTSize, int FFTsInBlock>
void* init_test(long long data_size, cudaStream_t stream) {
    const long long dim1 = FFTSize;
    const long long dim0 = data_size / dim1;

    cufftHandle* plan = new cufftHandle();

    checkCuFFT(cufftPlan1d(plan, dim1, CUFFT_C2C, dim0), "plan");
    checkCuFFT(cufftSetStream(*plan, stream), "cufftSetStream");

    return static_cast<void*>(plan);
}

template<int FFTSize, int FFTsInBlock>
void run_test(void* plan, cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems, cudaStream_t stream){
    checkCuFFT(cufftExecC2C(*static_cast<cufftHandle*>(plan), d_data, d_data, CUFFT_FORWARD), "exec");
}

// template<int FFTSize>
// void make_cufft_handle(cufftHandle* plan, long long data_size, cudaStream_t stream) {
//     const long long dim1 = FFTSize;
//     const long long dim0 = data_size / dim1;

//     checkCuFFT(cufftPlan1d(plan, dim1, CUFFT_C2C, dim0), "plan");
//     checkCuFFT(cufftSetStream(*plan, stream), "cufftSetStream");
// }

// template<int FFTSize, int FFTsInBlock>
// void exec_cufft_batch(cufftHandle plan, cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems, cudaStream_t stream) {
//     checkCuFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD), "exec");
// }