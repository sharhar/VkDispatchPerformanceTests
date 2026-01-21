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
#include <cstdint>
#include <cstring>
#include <complex>
#include <fstream>
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

float get_bandwith_scale_factor(long long elem_count, long long fft_size);

#define EXEC_MODE_CUFFT 1
#define EXEC_MODE_CUFFTDX 2
#define EXEC_MODE_CUFFTDX_NAIVE 3

template<int FFTSize, int FFTsInBlock, int exec_mode>
void* init_test(long long data_size, cudaStream_t stream);

template<int FFTSize, int FFTsInBlock, int exec_mode, bool validate>
void run_test(void* plan, cufftComplex* d_data, cufftComplex* d_kernel, long long total_elems, cudaStream_t stream);

template<int FFTSize, int FFTsInBlock, int exec_mode>
void delete_test(void* plan);

__global__ void fill_randomish(cufftComplex* a, long long n){
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        float x = __sinf(i * 0.00173f) + cosf(i * 0.00037f) + __tanf(i * 0.00029f) + 
                    sinhf(i * 0.013f) + coshf(i * 0.0019f) + tanhf(i * 0.00023f) + 
                    expf(sinf(i * 0.011f)) + expf(cosf(i * 0.007f));

        float y = __cosf(i * 0.00091f) + sinhf(i * 0.00053f) + tanhf(i * 0.00097f) + sinf(i * 0.23f) + coshf(i * 0.0037f) + tanf(i * 0.011f);
        a[i] = make_float2(cosf(x), sinf(y));
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
    int performance_cufftdx_naive;
    int performance_cufftdx;
    int performance_cufft;
    int validation;
    int warmup = 10;   // match Torch script's warmup
};

static Config parse_args(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <data_size> <iter_count> <iter_batch> <run_count> <run_type>\n";
        std::cerr << "  data_size   : total number of complex elements to process\n";
        std::cerr << "  iter_count  : total number of iterations to run\n";
        std::cerr << "  iter_batch  : number of iterations to batch into a single CUDA graph\n";
        std::cerr << "  run_count   : number of times to repeat the entire test for statistics\n";
        std::cerr << "  run_type    : 0 = validation, 1 = cufft performance, 2 = cufftdx performance, 3 = cufftdx naive performance\n";
        std::exit(1);
    }
    Config c;
    c.data_size  = std::stoll(argv[1]);
    c.iter_count = std::stoi(argv[2]);
    c.iter_batch = std::stoi(argv[3]);
    c.run_count  = std::stoi(argv[4]);
    int run_type  = std::stoi(argv[5]);
    c.validation = run_type == 0 ? 1 : 0;
    c.performance_cufft  = run_type == EXEC_MODE_CUFFT ? 1 : 0;
    c.performance_cufftdx = run_type == EXEC_MODE_CUFFTDX ? 1 : 0;
    c.performance_cufftdx_naive = run_type == EXEC_MODE_CUFFTDX_NAIVE ? 1 : 0;
    
    return c;
}

static double gb_per_exec(long long total_elems) {
    const double bytes = static_cast<double>(total_elems) * 8.0;
    return bytes / (1024.0 * 1024.0 * 1024.0);
}

template<int FFTSize, int FFTsInBlock, int exec_mode>
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
    void* plan = init_test<FFTSize, FFTsInBlock, exec_mode>(cfg.data_size, stream);

    // --- warmup on the stream (without graph) ---
    for (int i = 0; i < cfg.warmup; ++i)
        run_test<FFTSize, FFTsInBlock, exec_mode, false>(plan, d_data, d_kernel, cfg.data_size, stream);
    
    checkCuda(cudaStreamSynchronize(stream), "warmup sync");

    // === CUDA GRAPH CAPTURE ===
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    // Begin capturing the stream
    checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");

    // Capture iter_batch FFT operations into the graph
    for (int b = 0; b < cfg.iter_batch; ++b) {
        run_test<FFTSize, FFTsInBlock, exec_mode, false>(plan, d_data, d_kernel, cfg.data_size, stream);
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
    const double gb_per_exec_once = get_bandwith_scale_factor(cfg.data_size, FFTSize) * gb_per_exec(cfg.data_size);
    const double total_execs = static_cast<double>(cfg.iter_count);
    const double gb_per_second = (total_execs * gb_per_exec_once) / seconds;

    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);

    delete_test<FFTSize, FFTsInBlock, exec_mode>(plan);
    //cufftDestroy(plan);
    cudaStreamDestroy(stream);
    cudaFree(d_data);
    cudaFree(d_kernel);

    return gb_per_second;
}

template<int FFTSize, int FFTsInBlock, int exec_mode>
void do_fft_size_run(std::ofstream& out, const Config& cfg, const std::string& test_name) {
    std::vector<double> rates;
    rates.reserve(cfg.run_count);

    for (int r = 0; r < cfg.run_count; ++r) {
        const double gbps = run_cufft_case<FFTSize, FFTsInBlock, exec_mode>(cfg);
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

template<int exec_mode>
void run_performance_tests(const Config& cfg, const std::string& test_name) {
    const std::string output_name = test_name + ".csv";
    std::ofstream out(output_name);
    if (!out) {
        std::cerr << "Failed to open output file: " << output_name << "\n";
        return;
    }

    std::cout << "Running " << test_name << " tests with data size " << cfg.data_size
              << ", iter_count " << cfg.iter_count
              << ", iter_batch " << cfg.iter_batch
              << ", run_count " << cfg.run_count << "\n";

    // Header: Backend, FFT Size, Run 1..N, Mean, Std Dev
    out << "Backend,FFT Size";
    for (int i = 0; i < cfg.run_count; ++i) out << ",Run " << (i + 1) << " (GB/s)";
    out << ",Mean,Std Dev\n";

    do_fft_size_run<8, 256, exec_mode>(out, cfg, test_name);
    do_fft_size_run<16, 128, exec_mode>(out, cfg, test_name);
    do_fft_size_run<32, 64, exec_mode>(out, cfg, test_name);
    do_fft_size_run<64, 32, exec_mode>(out, cfg, test_name);
    do_fft_size_run<128, 32, exec_mode>(out, cfg, test_name);
    do_fft_size_run<256, 16, exec_mode>(out, cfg, test_name);
    do_fft_size_run<512, 16, exec_mode>(out, cfg, test_name);
    do_fft_size_run<1024, 8, exec_mode>(out, cfg, test_name);
    do_fft_size_run<2048, 4, exec_mode>(out, cfg, test_name);
    do_fft_size_run<4096, 2, exec_mode>(out, cfg, test_name);

    std::cout << "Results saved to " << output_name << "\n";
}

template<int FFTSize, int FFTsInBlock>
void run_validation_test(const Config& cfg) {
    cufftComplex* d_data = nullptr;
    checkCuda(cudaMalloc(&d_data, cfg.data_size * sizeof(cufftComplex)), "cudaMalloc d_data");
    checkCuda(cudaMemset(d_data, 0, cfg.data_size * sizeof(cufftComplex)), "cudaMemset d_data");

    cufftComplex* d_kernel = nullptr;
    checkCuda(cudaMalloc(&d_kernel, cfg.data_size * sizeof(cufftComplex)), "cudaMalloc d_kernel");
    checkCuda(cudaMemset(d_kernel, 0, cfg.data_size * sizeof(cufftComplex)), "cudaMemset d_kernel");

    cufftComplex* d_data_ref = nullptr;
    checkCuda(cudaMalloc(&d_data_ref, cfg.data_size * sizeof(cufftComplex)), "cudaMalloc d_data_ref");
    checkCuda(cudaMemset(d_data_ref, 0, cfg.data_size * sizeof(cufftComplex)), "cudaMemset d_data_ref");

    cufftComplex* d_kernel_ref = nullptr;
    checkCuda(cudaMalloc(&d_kernel_ref, cfg.data_size * sizeof(cufftComplex)), "cudaMalloc d_kernel_ref");
    checkCuda(cudaMemset(d_kernel_ref, 0, cfg.data_size * sizeof(cufftComplex)), "cudaMemset d_kernel_ref");

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

    // Copy to reference buffers
    cudaMemcpy(d_data_ref, d_data, cfg.data_size * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_kernel_ref, d_kernel, cfg.data_size * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);

    // --- Create explicit CUDA stream ---
    cudaStream_t stream;
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    // --- plan bound to the stream ---
    void* plan = init_test<FFTSize, FFTsInBlock, EXEC_MODE_CUFFT>(cfg.data_size, stream);
    void* plan_ref = init_test<FFTSize, FFTsInBlock, EXEC_MODE_CUFFTDX>(cfg.data_size, stream);

    // --- warmup on the stream (without graph) ---
    run_test<FFTSize, FFTsInBlock, EXEC_MODE_CUFFT, true>(plan, d_data, d_kernel, cfg.data_size, stream);
    run_test<FFTSize, FFTsInBlock, EXEC_MODE_CUFFTDX, true>(plan_ref, d_data_ref, d_kernel_ref, cfg.data_size, stream);

    cufftComplex* h_data = new cufftComplex[cfg.data_size];
    cufftComplex* h_data_ref = new cufftComplex[cfg.data_size];
    checkCuda(cudaMemcpy(h_data, d_data, cfg.data_size * sizeof(cufftComplex), cudaMemcpyDeviceToHost), "memcpy data");
    checkCuda(cudaMemcpy(h_data_ref, d_data_ref, cfg.data_size * sizeof(cufftComplex), cudaMemcpyDeviceToHost), "memcpy data ref");

    // Validate results
    int errors = 0;
    for (long long i = 0; i < cfg.data_size; ++i) {
        float diff_x = std::fabs(h_data[i].x - h_data_ref[i].x);
        float diff_y = std::fabs(h_data[i].y - h_data_ref[i].y);

        float diff_2 = diff_x * diff_x + diff_y * diff_y;

        float abs_2 = h_data_ref[i].x * h_data_ref[i].x + h_data_ref[i].y * h_data_ref[i].y;
        if (abs_2 < 1e-10f || diff_2 < 1e-10f) {
            // skip near-zero reference values
            continue;
        }

        if (diff_2 / abs_2 > 1e-6f) {
            if (errors < 10) {
                std::cout << "Mismatch at index " << i << ": got (" << h_data[i].x << ", " << h_data[i].y
                          << "), expected (" << h_data_ref[i].x << ", " << h_data_ref[i].y << ")"
                          << ", diff (" << diff_x << ", " << diff_y << ")"
                          << ", diff^2 " << diff_2 << ", diff^2/abs^2 " << diff_2 / abs_2 << "\n";
            }
            errors++;
        }
    }

    delete_test<FFTSize, FFTsInBlock, false>(plan);
    delete_test<FFTSize, FFTsInBlock, true>(plan_ref);

    //cufftDestroy(plan);
    cudaStreamDestroy(stream);
    cudaFree(d_data);
    cudaFree(d_kernel);
    cudaFree(d_data_ref);
    cudaFree(d_kernel_ref);
    delete[] h_data;
    delete[] h_data_ref;
    if (errors == 0) {
        std::cout << "Validation passed for FFT size " << FFTSize << "\n";
    } else {
        std::cout << "Validation failed for FFT size " << FFTSize << " with " << errors << " errors\n";
    }
}

void run_validation_tests(const Config& cfg) {
    std::cout << "Running validation tests with data size " << cfg.data_size << "\n";

    run_validation_test<8, 512>(cfg);
    run_validation_test<16, 256>(cfg);
    run_validation_test<32, 128>(cfg);
    run_validation_test<64, 64>(cfg);
    run_validation_test<128, 32>(cfg);
    run_validation_test<256, 16>(cfg);
    run_validation_test<512, 8>(cfg);
    run_validation_test<1024, 4>(cfg);
    run_validation_test<2048, 2>(cfg);
    run_validation_test<4096, 1>(cfg);
}

int main(int argc, char** argv) {
    const Config cfg = parse_args(argc, argv);

    if (cfg.validation)
        run_validation_tests(cfg);

    if (cfg.performance_cufftdx)
        run_performance_tests<EXEC_MODE_CUFFTDX>(cfg, "cufftdx");
    
    if (cfg.performance_cufft)
        run_performance_tests<EXEC_MODE_CUFFT>(cfg, "cufft");

    if (cfg.performance_cufftdx_naive)
        run_performance_tests<EXEC_MODE_CUFFTDX_NAIVE>(cfg, "cufftdx_naive");
    
    return 0;
}