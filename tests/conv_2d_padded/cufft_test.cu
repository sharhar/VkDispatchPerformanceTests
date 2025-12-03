#include "../common/common.cuh"

float get_bandwith_scale_factor() {
    return 11.0f;
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

__global__ void convolve_arrays(cufftComplex* data, cufftComplex* kernel, long long total_elems) {
    long long i = blockIdx.x * 1LL * blockDim.x + threadIdx.x;
    if (i < total_elems) {
        const size_t idx_in_image = i;
        const cufftComplex d = data[i];
        const cufftComplex k = kernel[idx_in_image];

        const float real = d.x * k.x - d.y * k.y;
        const float imag = d.x * k.y + d.y * k.x;
        data[i] = make_float2(real, imag);
    }
}

void exec_cufft_batch(cufftHandle plan, cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems, cudaStream_t stream) {
    checkCuFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD), "warmup");
    convolve_arrays<<<(total_elems+255)/256,256,0,stream>>>(d_data, d_kernel, total_elems);
    checkCuFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE), "warmup");
}