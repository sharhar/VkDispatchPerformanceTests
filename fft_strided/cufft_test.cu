#include "../common/common.cuh"

float get_bandwith_scale_factor() {
    return 2.0f;
}

void make_cufft_handle(cufftHandle* plan, long long data_size, int fft_size) {
    const long long dim2 = fft_size;
    const long long dim1 = fft_size;
    const long long dim0 = data_size / (dim1 * dim2);

    int n[1]        = { int(dim1) };
    int istride     = int(dim2);
    int ostride     = int(dim2);
    int idist       = 1;
    int odist       = 1;
    int batch       = int(dim0 * dim2);

    checkCuFFT(
        cufftPlanMany(plan, 1, n,
                        nullptr, istride, idist,
                        nullptr, ostride, odist,
                        CUFFT_C2C,
                        batch),
        "plan_1d_axis1");
}

void exec_cufft_batch(cufftHandle plan, cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems) {
    checkCuFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD), "exec");
}