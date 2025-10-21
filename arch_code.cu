/*

Simple CUDA program that prionts to stdout the compute capability of a given GPU
as two digits, e.g., "86" for compute capability 8.6.

*/
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

int main(int argc, char** argv) {
    int dev = (argc > 1) ? std::atoi(argv[1]) : 0;

    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess || count == 0) {
        std::fprintf(stderr, "No CUDA devices found: %s\n", cudaGetErrorString(e));
        return 1;
    }
    if (dev < 0 || dev >= count) {
        std::fprintf(stderr, "Invalid device index %d (0..%d)\n", dev, count - 1);
        return 1;
    }

    cudaDeviceProp prop{};
    e = cudaGetDeviceProperties(&prop, dev);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(e));
        return 1;
    }

    std::printf("%d%d\n", prop.major, prop.minor); // e.g., 86
    return 0;
}
