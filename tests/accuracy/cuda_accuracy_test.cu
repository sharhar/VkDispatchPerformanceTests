#include <cuda_runtime_api.h>
#include <cufft.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifndef CUDA_CHECK_AND_EXIT
#    define CUDA_CHECK_AND_EXIT(error)                                                                      \
        {                                                                                                   \
            auto status = static_cast<cudaError_t>(error);                                                  \
            if (status != cudaSuccess) {                                                                    \
                std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                                          \
            }                                                                                               \
        }
#endif // CUDA_CHECK_AND_EXIT

#include "../common/nonstrided_kernels.cuh"

namespace {

constexpr int EXEC_MODE_CUFFT = 1;
constexpr int EXEC_MODE_CUFFTDX = 2;

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] " << what << " failed: " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

void check_cufft(cufftResult err, const char* what) {
    if (err != CUFFT_SUCCESS) {
        std::cerr << "[cuFFT] " << what << " failed with error code " << err << "\n";
        std::exit(1);
    }
}

bool read_binary_file(const std::string& filename, std::vector<cufftComplex>* values) {
    std::ifstream input(filename, std::ios::binary | std::ios::ate);
    if (!input) {
        std::cerr << "Failed to open input file: " << filename << "\n";
        return false;
    }

    const std::streamsize byte_count = input.tellg();
    if (byte_count < 0) {
        std::cerr << "Failed to read input file size: " << filename << "\n";
        return false;
    }

    if ((byte_count % static_cast<std::streamsize>(sizeof(cufftComplex))) != 0) {
        std::cerr << "Input file size is not aligned to complex64 values: " << filename << "\n";
        return false;
    }

    const std::size_t value_count = static_cast<std::size_t>(byte_count / sizeof(cufftComplex));
    values->resize(value_count);

    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(values->data()), byte_count);

    if (!input) {
        std::cerr << "Failed to read input file contents: " << filename << "\n";
        return false;
    }

    return true;
}

bool write_binary_file(const std::string& filename, const std::vector<cufftComplex>& values) {
    std::ofstream output(filename, std::ios::binary);
    if (!output) {
        std::cerr << "Failed to open output file: " << filename << "\n";
        return false;
    }

    output.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(cufftComplex));
    if (!output) {
        std::cerr << "Failed to write output file: " << filename << "\n";
        return false;
    }

    return true;
}

template<int FFTSize, int FFTsInBlock, int exec_mode>
void run_fft_for_size(cufftComplex* d_data, long long data_size, cudaStream_t stream) {
    if constexpr (exec_mode == EXEC_MODE_CUFFT) {
        if ((data_size % FFTSize) != 0) {
            std::cerr << "data_size must be divisible by FFT size for cuFFT backend\n";
            std::exit(1);
        }

        cufftHandle plan;
        check_cufft(cufftPlan1d(&plan, FFTSize, CUFFT_C2C, static_cast<int>(data_size / FFTSize)), "cufftPlan1d");
        check_cufft(cufftSetStream(plan, stream), "cufftSetStream");
        check_cufft(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD), "cufftExecC2C");
        check_cufft(cufftDestroy(plan), "cufftDestroy");
        return;
    }

    if ((data_size % (static_cast<long long>(FFTSize) * FFTsInBlock)) != 0) {
        std::cerr << "data_size must be divisible by FFTSize * FFTsInBlock for cuFFTDx backend\n";
        std::exit(1);
    }

    auto config = NonStridedFFTConfig<FFTSize, FFTsInBlock, 1>();
    config.init(stream);
    config.execute_fft(d_data, data_size, stream);
}

template<int exec_mode>
bool run_fft(int fft_size, cufftComplex* d_data, long long data_size, cudaStream_t stream) {
    switch (fft_size) {
        case 8:
            run_fft_for_size<8, 256, exec_mode>(d_data, data_size, stream);
            return true;
        case 16:
            run_fft_for_size<16, 128, exec_mode>(d_data, data_size, stream);
            return true;
        case 32:
            run_fft_for_size<32, 64, exec_mode>(d_data, data_size, stream);
            return true;
        case 64:
            run_fft_for_size<64, 32, exec_mode>(d_data, data_size, stream);
            return true;
        case 128:
            run_fft_for_size<128, 32, exec_mode>(d_data, data_size, stream);
            return true;
        case 256:
            run_fft_for_size<256, 16, exec_mode>(d_data, data_size, stream);
            return true;
        case 512:
            run_fft_for_size<512, 16, exec_mode>(d_data, data_size, stream);
            return true;
        case 1024:
            run_fft_for_size<1024, 8, exec_mode>(d_data, data_size, stream);
            return true;
        case 2048:
            run_fft_for_size<2048, 4, exec_mode>(d_data, data_size, stream);
            return true;
        case 4096:
            run_fft_for_size<4096, 2, exec_mode>(d_data, data_size, stream);
            return true;
        default:
            return false;
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <backend:cufft|cufftdx> <fft_size> <data_size> <input_file> <output_file>\n";
        return 1;
    }

    const std::string backend = argv[1];
    const int fft_size = std::stoi(argv[2]);
    const long long data_size = std::stoll(argv[3]);
    const std::string input_file = argv[4];
    const std::string output_file = argv[5];

    if (fft_size <= 0 || data_size <= 0) {
        std::cerr << "fft_size and data_size must be positive\n";
        return 1;
    }

    std::vector<cufftComplex> host_data;
    if (!read_binary_file(input_file, &host_data)) {
        return 1;
    }

    if (host_data.size() != static_cast<std::size_t>(data_size)) {
        std::cerr << "Input size mismatch. Expected " << data_size
                  << " complex elements, got " << host_data.size() << "\n";
        return 1;
    }

    cufftComplex* d_data = nullptr;
    const std::size_t data_bytes = host_data.size() * sizeof(cufftComplex);
    check_cuda(cudaMalloc(&d_data, data_bytes), "cudaMalloc");
    check_cuda(cudaMemcpy(d_data, host_data.data(), data_bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    bool ran = false;
    if (backend == "cufft") {
        ran = run_fft<EXEC_MODE_CUFFT>(fft_size, d_data, data_size, stream);
    } else if (backend == "cufftdx") {
        ran = run_fft<EXEC_MODE_CUFFTDX>(fft_size, d_data, data_size, stream);
    } else {
        std::cerr << "Unsupported backend: " << backend << "\n";
    }

    if (!ran) {
        cudaStreamDestroy(stream);
        cudaFree(d_data);
        if (backend == "cufft" || backend == "cufftdx") {
            std::cerr << "Unsupported FFT size: " << fft_size << "\n";
        }
        return 1;
    }

    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    check_cuda(cudaMemcpy(host_data.data(), d_data, data_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    check_cuda(cudaFree(d_data), "cudaFree");

    if (!write_binary_file(output_file, host_data)) {
        return 1;
    }

    return 0;
}
