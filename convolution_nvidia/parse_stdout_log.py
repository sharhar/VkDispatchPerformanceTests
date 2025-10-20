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

            cufftdx_line = lines_list[ii + 7]
            cufft_line = lines_list[ii + 16]

            assert cufftdx_line.startswith('Time (all) [ms_n]: ')
            assert cufft_line.startswith('Time (all) [ms_n]: ')
            
            prefix_size = len('Time (all) [ms_n]: ')

            cufftdx_time = float(cufftdx_line[prefix_size:].strip())
            cufft_time = float(cufft_line[prefix_size:].strip())

            cufft_data[(iter_count, fft_size)] = cufft_time
            cufftdx_data[(iter_count, fft_size)] = cufftdx_time

            max_iter_count = max(max_iter_count, iter_count)

            #print(ii, iter_count, fft_size)
            #print("CufftDx time: ", cufftdx_time)
            #print("Cufft time: ", cufft_time)

run_str = "".join([f"Run {i+1} (ms)," for i in range(max_iter_count)])

fft_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

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

with open('test_results/cufft.csv', 'w') as f:
    f.write(cufft_header)

with open('test_results/cufftdx.csv', 'w') as f:
    f.write(cufftdx_header)

# with open('test_results/nvidia_ratios.csv', 'w') as f:
#     f.write(ratios_header)