import sys
import numpy as np

filename = sys.argv[1]

max_iter_count = 0

cufft_data = {}
cufftdx_data = {}

with open(filename, 'r') as f:
    full_text = f.read()

    lines_list = list(full_text.splitlines())

    for ii, line in enumerate(lines_list):
        correct_start = line.startswith('Iteration ')
        correct_fft_index = line.find('fft_size=')

        if correct_start and correct_fft_index != -1:
            parts = line.split('|')

            iter_count = int(parts[0].strip().split(' ')[1])
            fft_size = int(parts[1].strip().split('=')[1])

            fft_count_line = lines_list[ii + 3]
            cufftdx_line = lines_list[ii + 6]
            cufft_line = lines_list[ii + 15]

            assert fft_count_line.startswith('FFTs run: ')
            assert cufftdx_line.startswith('Avg Time [ms_n]: ')
            assert cufft_line.startswith('Avg Time [ms_n]: ')
            
            fft_count = int(fft_count_line[len('FFTs run: '):].strip())

            gb_byte_count = fft_size * 2 * 4 * fft_count / (1 << 30)

            data_size = 6 * gb_byte_count # We read and write 3 times

            prefix_size = len('Avg Time [ms_n]: ')

            cufftdx_time = float(cufftdx_line[prefix_size:].strip())
            cufft_time = float(cufft_line[prefix_size:].strip())

            cufftdx_bandwidth = data_size / (cufftdx_time * 1e-3)
            cufft_bandwitdh = data_size / (cufft_time * 1e-3)

            cufft_data[(iter_count, fft_size)] = cufft_bandwitdh
            cufftdx_data[(iter_count, fft_size)] = cufftdx_bandwidth

            max_iter_count = max(max_iter_count, iter_count)

run_str = "".join([f"Run {i+1} (GB/s)," for i in range(max_iter_count)])

fft_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

cufft_header = "Backend,FFT Size," + run_str + "Mean,Std Dev\n"
cufftdx_header = "Backend,FFT Size," + run_str + "Mean,Std Dev\n"

ratios_header = "Backend,FFT Size,Ratio,Std Dev\n"

for fft_size in fft_sizes:
    cufft_iterations = []
    cufftdx_iterations = []

    for i in range(max_iter_count):
        cufft_iterations.append(cufft_data[(i + 1, fft_size)])
        cufftdx_iterations.append(cufftdx_data[(i + 1, fft_size)])

    cufft_mean = np.mean(cufft_iterations)
    cufft_std = np.std(cufft_iterations)

    cufftdx_mean = np.mean(cufftdx_iterations)
    cufftdx_std = np.std(cufftdx_iterations)

    cufft_data_str = ",".join([f"{x:.4f}" for x in cufft_iterations])
    cufftdx_data_str = ",".join([f"{x:.4f}" for x in cufftdx_iterations])

    cufft_header += f"cufft,{fft_size}," + cufft_data_str + f",{cufft_mean:.4f},{cufft_std:.4f}\n"
    cufftdx_header += f"cufftdx,{fft_size}," + cufftdx_data_str + f",{cufftdx_mean:.4f},{cufftdx_std:.4f}\n"

    ratio = cufft_mean / cufftdx_mean

    ratio_std = ratio * np.sqrt( (cufft_std / cufft_mean)**2 + (cufftdx_std / cufftdx_mean)**2 )

    ratios_header += f"nvidia,{fft_size},{ratio:.4f},{ratio_std:.4f}\n"

with open('test_results/cufft_nvidia.csv', 'w') as f:
    f.write(cufft_header)

with open('test_results/cufftdx_nvidia.csv', 'w') as f:
    f.write(cufftdx_header)