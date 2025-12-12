#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include <cufft.h>
#include <cufft.h>
#include <vector>
#include <iostream>
#include <cmath>

// Helper to compare two complex arrays
bool compare_results(float2* d_result_test, float2* d_result_ref, size_t n, float epsilon = 1e-3) {
    std::vector<float2> h_test(n);
    std::vector<float2> h_ref(n);

    cudaMemcpy(h_test.data(), d_result_test, n * sizeof(float2), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref.data(), d_result_ref, n * sizeof(float2), cudaMemcpyDeviceToHost);

    float max_error = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff_real = h_test[i].x - h_ref[i].x;
        float diff_imag = h_test[i].y - h_ref[i].y;
        float err = sqrtf(diff_real*diff_real + diff_imag*diff_imag);
        
        float magnitude = sqrtf(h_ref[i].x*h_ref[i].x + h_ref[i].y*h_ref[i].y);
        
        if (magnitude > 1e-6) err /= magnitude; // Relative error
        if (err > max_error) max_error = err;
    }

    std::cout << "Max Relative Error: " << max_error << std::endl;
    return max_error < epsilon;
}

// Wrapper for cuFFT Validation
bool run_cufft_validation(cufftComplex* d_input, cufftComplex* d_output_test, int N, int batch_size) {
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, batch_size);

    // Allocate reference output
    cufftComplex* d_output_ref;
    cudaMalloc(&d_output_ref, N * batch_size * sizeof(cufftComplex));

    // Run cuFFT (Golden Standard)
    // Note: cuFFT can be in-place or out-of-place. 
    // Ensure d_input matches the state expected by your test kernel.
    cufftExecC2C(plan, d_input, d_output_ref, CUFFT_FORWARD);
    cudaDeviceSynchronize();

    bool pass = compare_results(d_output_test, d_output_ref, N * batch_size);

    cufftDestroy(plan);
    cudaFree(d_output_ref);
    return pass;
}