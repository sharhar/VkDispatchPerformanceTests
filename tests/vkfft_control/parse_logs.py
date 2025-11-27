import csv

def extract_data_from_log(log_file_path):
    data = {}
    with open(log_file_path, 'r') as file:
        for line in file:
            if not line.startswith("VkFFT") and not line.startswith("cuFFT"):
                continue

            parts = line.strip().split(' ')

            fft_size = int(parts[3].split("x")[0])

            if fft_size < 64 or fft_size > 4096:
                continue

            exec_time = float(parts[8])
            exec_time_std = float(parts[11])

            gbps = 4 / (exec_time * 1e-3)
            gbps_std = gbps * (exec_time_std / exec_time)

            data[fft_size] = (gbps, gbps_std)
    return data

def write_data_to_csv(backend, data, csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Backend', 'FFT Size', 'GB/s', 'GB/s Std Dev'])
        for fft_size in sorted(data.keys()):
            gbps, gbps_std = data[fft_size]

            gbps = round(gbps, 3)
            gbps_std = round(gbps_std, 3)

            csv_writer.writerow([backend, fft_size, gbps, gbps_std])

vulkan_data = extract_data_from_log("vkfft_control_vulkan_output.log")
cuda_data = extract_data_from_log("vkfft_control_cuda_output.log")
cufft_data = extract_data_from_log("vkfft_control_cufft_output.log")

write_data_to_csv("vulkan", vulkan_data, "test_results/vulkan_performance.csv")
write_data_to_csv("cuda", cuda_data, "test_results/cuda_performance.csv")
write_data_to_csv("cufft", cufft_data, "test_results/cufft_performance.csv")