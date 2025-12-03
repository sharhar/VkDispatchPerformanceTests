#include "../common/common.cuh"

float get_bandwith_scale_factor() {
    return 4.0f;
}

void make_cufft_handle(cufftHandle* plan, long long data_size, int fft_size, cudaStream_t stream) {
    const long long dim2 = fft_size;
    const long long dim1 = fft_size;
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
}

void exec_cufft_batch(cufftHandle plan, cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems, cudaStream_t stream) {
    checkCuFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD), "exec");
}