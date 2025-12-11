#include "../common/common.cuh"

const char* get_test_name() {
    return "cufft";
}

float get_bandwith_scale_factor() {
    return 4.0f;
}

template<int FFTSize, int FFTsInBlock>
void* init_test(long long data_size, cudaStream_t stream) {
    cufftHandle* plan = new cufftHandle();

    const long long dim2 = FFTSize;
    const long long dim1 = FFTSize;
    const long long dim0 = data_size / (dim1 * dim2);

    int n[2] = { int(dim1), int(dim2) };
    int inembed[2] = { int(dim1), int(dim2) };
    int onembed[2] = { int(dim1), int(dim2) };
    int istride    = 1;
    int ostride    = 1;
    int idist      = int(dim1)* int(dim2);
    int odist      = int(dim1)* int(dim2);

    checkCuFFT(cufftPlanMany(plan, 2, n,
                                  inembed,  istride, idist,
                                  onembed,  ostride, odist,
                                  CUFFT_C2C, int(dim0)), "plan2d");
    checkCuFFT(cufftSetStream(*plan, stream), "cufftSetStream");

    return static_cast<void*>(plan);
}

template<int FFTSize, int FFTsInBlock>
void run_test(void* plan, cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems, cudaStream_t stream){
    checkCuFFT(cufftExecC2C(*static_cast<cufftHandle*>(plan), d_data, d_data, CUFFT_FORWARD), "exec");
}