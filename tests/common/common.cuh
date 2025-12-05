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

const char* get_test_name();

float get_bandwith_scale_factor();

template<int FFTSize>
void make_cufft_handle(cufftHandle* plan, long long data_size, int fft_size, cudaStream_t stream);

template<int FFTSize, int FFTsInBlock>
void exec_cufft_batch(cufftHandle plan, cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems, cudaStream_t stream);

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
    int warmup = 10;   // match Torch script's warmup
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

static double gb_per_exec(long long total_elems) {
    const double bytes = static_cast<double>(total_elems) * 8.0;
    return bytes / (1024.0 * 1024.0 * 1024.0);
}

template<int FFTSize, int FFTsInBlock>
static double run_cufft_case(const Config& cfg) {

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

    // --- Create explicit CUDA stream ---
    cudaStream_t stream;
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    // --- plan bound to the stream ---
    cufftHandle plan;
    make_cufft_handle<FFTSize>(&plan, cfg.data_size, stream);

    // --- warmup on the stream (without graph) ---
    for (int i = 0; i < cfg.warmup; ++i)
        exec_cufft_batch<FFTSize, FFTsInBlock>(plan, d_data, d_kernel, cfg.data_size, stream);
    
    checkCuda(cudaStreamSynchronize(stream), "warmup sync");

    // === CUDA GRAPH CAPTURE ===
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    // Begin capturing the stream
    checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");

    // Capture iter_batch FFT operations into the graph
    for (int b = 0; b < cfg.iter_batch; ++b) {
        exec_cufft_batch<FFTSize, FFTsInBlock>(plan, d_data, d_kernel, cfg.data_size, stream);
    }

    // End capture and obtain the graph
    checkCuda(cudaStreamEndCapture(stream, &graph), "end capture");

    // Instantiate the graph into an executable form
    checkCuda(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0), "graph instantiate");

    // Calculate how many times to launch the graph
    const int num_graph_launches = (cfg.iter_count + cfg.iter_batch - 1) / cfg.iter_batch;

    // === TIMED EXECUTION ===
    cudaEvent_t evA, evB;
    checkCuda(cudaEventCreate(&evA), "evA");
    checkCuda(cudaEventCreate(&evB), "evB");
    
    checkCuda(cudaEventRecord(evA, stream), "record A");
    
    // Launch the graph multiple times to reach iter_count
    for (int i = 0; i < num_graph_launches; ++i) {
        checkCuda(cudaGraphLaunch(graphExec, stream), "graph launch");
    }
    
    checkCuda(cudaEventRecord(evB, stream), "record B");
    checkCuda(cudaEventSynchronize(evB), "sync B");
    
    float ms = 0.f;
    checkCuda(cudaEventElapsedTime(&ms, evA, evB), "elapsed");
    checkCuda(cudaEventDestroy(evA), "dA");
    checkCuda(cudaEventDestroy(evB), "dB");

    // Convert elapsed to seconds
    const double seconds = static_cast<double>(ms) / 1000.0;

    // Compute throughput in GB/s
    const double gb_per_exec_once = get_bandwith_scale_factor() * gb_per_exec(cfg.data_size);
    const double total_execs = static_cast<double>(cfg.iter_count);
    const double gb_per_second = (total_execs * gb_per_exec_once) / seconds;

    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cufftDestroy(plan);
    cudaStreamDestroy(stream);
    cudaFree(d_data);
    cudaFree(d_kernel);

    return gb_per_second;
}

template<int FFTSize, int FFTsInBlock>
void do_fft_size_run(std::ofstream& out, const Config& cfg, const std::string& test_name) {
    std::vector<double> rates;
    rates.reserve(cfg.run_count);

    for (int r = 0; r < cfg.run_count; ++r) {
        const double gbps = run_cufft_case<FFTSize, FFTsInBlock>(cfg);
        std::cout << "FFT Size: " << FFTSize << ", Throughput: " << std::fixed << std::setprecision(4)
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
    out << test_name << "," << FFTSize;
    out << std::fixed << std::setprecision(4);
    for (double v : rates) out << "," << v;
    out << "," << mean << "," << stdev << "\n";
}

int main(int argc, char** argv) {
    const Config cfg = parse_args(argc, argv);

    const std::string test_name = get_test_name(); 

    const std::string output_name = test_name + ".csv";
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

    do_fft_size_run<8, 256>(out, cfg, test_name);
    do_fft_size_run<16, 128>(out, cfg, test_name);
    do_fft_size_run<32, 64>(out, cfg, test_name);
    do_fft_size_run<64, 32>(out, cfg, test_name);
    do_fft_size_run<128, 32>(out, cfg, test_name);
    do_fft_size_run<256, 32>(out, cfg, test_name);
    do_fft_size_run<512, 16>(out, cfg, test_name);
    do_fft_size_run<1024, 8>(out, cfg, test_name);
    do_fft_size_run<2048, 4>(out, cfg, test_name);
    do_fft_size_run<4096, 2>(out, cfg, test_name);

    std::cout << "Results saved to " << output_name << "\n";
    return 0;
}