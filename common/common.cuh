#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

float get_bandwith_scale_factor();
void make_cufft_handle(cufftHandle* plan, long long data_size, int fft_size);
void exec_cufft_batch(cufftHandle plan, cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems);

__global__ void fill_randomish(cufftComplex* a, long long n){
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        float x = __sinf(i * 0.00173f);
        float y = __cosf(i * 0.00091f);
        a[i] = make_float2(x, y);
    }
}

static inline void checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] " << what << " failed: " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

static inline void checkCuFFT(cufftResult err, const char* what) {
    if (err != CUFFT_SUCCESS) {
        std::cerr << "[cuFFT] " << what << " failed: " << err << "\n";
        std::exit(1);
    }
}

struct Config {
    long long data_size;
    int iter_count;
    int iter_batch;
    int run_count;
    int warmup = 10;   // match Torch script’s warmup
};

static Config parse_args(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <data_size> <iter_count> <iter_batch> <run_count>\n";
        std::exit(1);
    }
    Config c;
    c.data_size  = std::stoll(argv[1]);
    c.iter_count = std::stoi(argv[2]);
    c.iter_batch = std::stoi(argv[3]);
    c.run_count  = std::stoi(argv[4]);
    return c;
}

static std::vector<int> get_fft_sizes() {
    std::vector<int> sizes;
    for (int p = 6; p <= 12; ++p) sizes.push_back(1 << p); // 64..4096
    return sizes;
}

static double gb_per_exec(long long total_elems) {
    const double bytes = static_cast<double>(total_elems) * 8.0;
    return bytes / (1024.0 * 1024.0 * 1024.0);
}

static double run_cufft_case(const Config& cfg, int fft_size) {

    cufftComplex* d_data = nullptr;
    checkCuda(cudaMalloc(&d_data, cfg.data_size * sizeof(cufftComplex)), "cudaMalloc d_data");
    checkCuda(cudaMemset(d_data, 0, cfg.data_size * sizeof(cufftComplex)), "cudaMemset d_data");

    cufftComplex* d_kernel = nullptr;
    checkCuda(cudaMalloc(&d_kernel, cfg.data_size * sizeof(cufftComplex)), "cudaMalloc d_kernel");
    checkCuda(cudaMemset(d_kernel, 0, cfg.data_size * sizeof(cufftComplex)), "cudaMemset d_kernel");


    {
        int t = 256, b = int((cfg.data_size + t - 1) / t);
        fill_randomish<<<b,t>>>(d_data, cfg.data_size);
        checkCuda(cudaGetLastError(), "fill launch");
        checkCuda(cudaDeviceSynchronize(), "fill sync");

        int kt = 256, kb = int((cfg.data_size + kt - 1) / kt);
        fill_randomish<<<kb,kt>>>(d_kernel, cfg.data_size);
        checkCuda(cudaGetLastError(), "fill kernel launch");
        checkCuda(cudaDeviceSynchronize(), "fill kernel sync");
    }

    // --- plan bound to the stream ---
    cufftHandle plan;
    make_cufft_handle(&plan, cfg.data_size, fft_size);

    // --- warmup on the stream ---
    for (int i = 0; i < cfg.warmup; ++i)
        exec_cufft_batch(plan, d_data, d_kernel, cfg.data_size);

    //checkCuFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD), "warmup");
    
    checkCuda(cudaDeviceSynchronize(), "warmup sync");

    // === OPTION A: plain single-stream timing (simple & robust) ===
    cudaEvent_t evA, evB;
    checkCuda(cudaEventCreate(&evA), "evA");
    checkCuda(cudaEventCreate(&evB), "evB");
    checkCuda(cudaEventRecord(evA), "record A");
    for (int it = 0; it < cfg.iter_count; ++it)
        exec_cufft_batch(plan, d_data, d_kernel, cfg.data_size);
        //checkCuFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD), "exec");
    checkCuda(cudaEventRecord(evB), "record B");
    checkCuda(cudaEventSynchronize(evB), "sync B");
    checkCuda(cudaDeviceSynchronize(), "warmup sync");
    float ms = 0.f; checkCuda(cudaEventElapsedTime(&ms, evA, evB), "elapsed");
    checkCuda(cudaEventDestroy(evA), "dA");
    checkCuda(cudaEventDestroy(evB), "dB");

    // Convert elapsed to seconds
    const double seconds = static_cast<double>(ms) / 1000.0;

    // Compute throughput in GB/s (same accounting as Torch: 2 * elems * 8 bytes per exec)
    const double gb_per_exec_once = get_bandwith_scale_factor() * gb_per_exec(cfg.data_size);
    const double total_execs = static_cast<double>(cfg.iter_count); // * static_cast<double>(cfg.iter_batch);
    const double gb_per_second = (total_execs * gb_per_exec_once) / seconds;

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_data);

    return gb_per_second;
}

int main(int argc, char** argv) {
    const Config cfg = parse_args(argc, argv);
    const auto sizes = get_fft_sizes();

    const std::string output_name = "cufft.csv";
    std::ofstream out(output_name);
    if (!out) {
        std::cerr << "Failed to open output file: " << output_name << "\n";
        return 1;
    }

    std::cout << "Running cuFFT tests with data size " << cfg.data_size
              << ", iter_count " << cfg.iter_count
              << ", iter_batch " << cfg.iter_batch
              << ", run_count " << cfg.run_count << "\n";

    // Header: Backend, FFT Size, Run 1..N, Mean, Std Dev
    out << "Backend,FFT Size";
    for (int i = 0; i < cfg.run_count; ++i) out << ",Run " << (i + 1) << " (GB/s)";
    out << ",Mean,Std Dev\n";

    for (int fft_size : sizes) {
        std::vector<double> rates;
        rates.reserve(cfg.run_count);

        for (int r = 0; r < cfg.run_count; ++r) {
            const double gbps = run_cufft_case(cfg, fft_size);
            std::cout << "FFT Size: " << fft_size << ", Throughput: " << std::fixed << std::setprecision(2)
                      << gbps << " GB/s\n";
            rates.push_back(gbps);
        }

        // Compute mean/std
        double mean = 0.0;
        for (double v : rates) mean += v;
        mean /= static_cast<double>(rates.size());

        double var = 0.0;
        for (double v : rates) {
            const double d = v - mean;
            var += d * d;
        }
        var /= static_cast<double>(rates.size());
        const double stdev = std::sqrt(var);

        // Round to 2 decimals like your Torch script
        out << "cufft," << fft_size;
        out << std::fixed << std::setprecision(2);
        for (double v : rates) out << "," << v;
        out << "," << mean << "," << stdev << "\n";
    }

    std::cout << "Results saved to " << output_name << "\n";
    return 0;
}
