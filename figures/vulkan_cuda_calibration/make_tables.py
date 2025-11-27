import os
import csv

from typing import Dict, Tuple, Set

def get_test_data(filename: str) -> Dict[int, Tuple[float, float]]:
    results = {}

    with open(filename, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            size = int(row['FFT Size'])
            mean = float(row['Mean'])
            std = float(row['Std Dev'])
            results[size] = (mean, std)
    
    return results

cufft_data = get_test_data('../../test_results/vkfft_control/cufft.csv')
cuda_data = get_test_data('../../test_results/vkfft_control/cuda.csv')
vulkan_data = get_test_data('../../test_results/vkfft_control/vulkan.csv')

latex_table = """\\begin{table}[htbp]
\\caption{Comparison of Performance (GB/s) for Batched 1D FFTs from VkFFT's benchmark}
\\begin{center}
\\begin{tabular}{c c c c}
\\hline
\\textbf{FFT Size ($N$)} & \\textbf{cuFFT} & \\textbf{VkFFT (CUDA)} & \\textbf{VkFFT (Vulkan)} \\\\
\\hline
"""

sum_cufft = 0
sum_cuda = 0
sum_vulkan = 0

std_sum_cufft = 0
std_sum_cuda = 0
std_sum_vulkan = 0

for size in sorted(cufft_data.keys()):
    cufft_mean, cufft_std = cufft_data[size]
    cuda_mean, cuda_std = cuda_data[size]
    vulkan_mean, vulkan_std = vulkan_data[size]

    sum_cufft += cufft_mean
    sum_cuda += cuda_mean
    sum_vulkan += vulkan_mean

    std_sum_cuda += cuda_std ** 2
    std_sum_vulkan += vulkan_std ** 2
    std_sum_cufft += cufft_std ** 2

    latex_table += f"{size} & {cufft_mean:.1f} $\\pm$ {cufft_std:.3f} & {cuda_mean:.2f} $\\pm$ {cuda_std:.4f} & {vulkan_mean:.2f} $\\pm$ {vulkan_std:.4f} \\\\\n"

latex_table += """\\hline
\\end{tabular}
\\label{tab:vkfft_control}
\\end{center}
\\end{table}"""

latex_table += f"\nAverage Performance: & {sum_cufft / len(cufft_data):.2f} & {sum_cuda / len(cuda_data):.2f} & {sum_vulkan / len(vulkan_data):.2f} \\\\\n"

std_sum_cuda = std_sum_cuda ** 0.5
std_sum_vulkan = std_sum_vulkan ** 0.5
std_sum_cufft = std_sum_cufft ** 0.5

latex_table += f"Average Std Dev: & {std_sum_cufft / len(cufft_data):.4f} & {std_sum_cuda / len(cuda_data):.4f} & {std_sum_vulkan / len(vulkan_data):.4f} \\\\\n"

ratio = sum_vulkan / sum_cuda
ratio_std = ratio * ((std_sum_cuda / sum_cuda) ** 2 + (std_sum_vulkan / sum_vulkan) ** 2) ** 0.5

ratio -= 1
ratio *= 100
ratio_std *= 100

latex_table += f"\nAverage VkFFT (Vulkan) / VkFFT (CUDA) Ratio: & \\multicolumn{{3}}{{c}}{{{ratio:.3f} $\\pm$ {ratio_std:.4f}}} \\\\\n"

with open('nvidia_control.tex', 'w') as f:
    f.write(latex_table)

cufft_data = get_test_data('../../test_results/fft_nonstrided/cufft.csv')
cufftdx_data = get_test_data('../../test_results/fft_nonstrided/zipfft.csv')
vkdispatch_data = get_test_data('../../test_results/fft_nonstrided/vkdispatch.csv')
vkfft_data = get_test_data('../../test_results/fft_nonstrided/vkfft.csv')

latex_table = """\\begin{table}[htbp]
\\caption{Comparison of Performance (GB/s) for Batched 1D FFTs from our benchmark}
\\begin{center}
\\begin{tabular}{c c c c}
\\hline
\\textbf{FFT Size ($N$)} & \\textbf{cuFFT} & \\textbf{cuFFTdx} & \\textbf{VkFFT (Vulkan)} & \\textbf{vkdispatch} \\\\
\\hline
"""

sum_cufft = 0
sum_cufftdx = 0
sum_vkdispatch = 0
sum_vkfft = 0

std_sum_cufft = 0
std_sum_cufftdx = 0
std_sum_vkdispatch = 0
std_sum_vkfft = 0

for size in sorted(cufft_data.keys()):
    cufft_mean, cufft_std = cufft_data[size]
    cufftdx_mean, cufftdx_std = cufftdx_data[size]
    vkdispatch_mean, vkdispatch_std = vkdispatch_data[size]
    vkfft_mean, vkfft_std = vkfft_data[size]

    sum_cufft += cufft_mean
    sum_cufftdx += cufftdx_mean
    sum_vkdispatch += vkdispatch_mean
    sum_vkfft += vkfft_mean

    std_sum_cufft += cufft_std ** 2
    std_sum_cufftdx += cufftdx_std ** 2
    std_sum_vkdispatch += vkdispatch_std ** 2
    std_sum_vkfft += vkfft_std ** 2

    latex_table += f"{size} & {cufft_mean:.1f} $\\pm$ {cufft_std:.3f} & {cufftdx_mean:.2f} $\\pm$ {cufftdx_std:.4f} & {vkfft_mean:.2f} $\\pm$ {vkfft_std:.4f} & {vkdispatch_mean:.2f} $\\pm$ {vkdispatch_std:.4f} \\\\\n"

latex_table += """\\hline
\\end{tabular}
\\label{tab:1d_fft}
\\end{center}
\\end{table}"""

latex_table += f"\nAverage Performance: & {sum_cufft / len(cufft_data):.2f} & {sum_cufftdx / len(cufftdx_data):.2f} & {sum_vkfft / len(vkfft_data):.2f} & {sum_vkdispatch / len(vkdispatch_data):.2f} \\\\\n"

std_sum_cufft = std_sum_cufft ** 0.5
std_sum_cufftdx = std_sum_cufftdx ** 0.5
std_sum_vkdispatch = std_sum_vkdispatch ** 0.5
std_sum_vkfft = std_sum_vkfft ** 0.5

latex_table += f"Average Std Dev: & {std_sum_cufft / len(cufft_data):.4f} & {std_sum_cufftdx / len(cufftdx_data):.4f} & {std_sum_vkfft / len(vkfft_data):.4f} & {std_sum_vkdispatch / len(vkdispatch_data):.4f} \\\\\n"

#ratio = sum_vulkan / sum_cuda
#ratio_std = ratio * ((std_sum_cuda / sum_cuda) ** 2 + (std_sum_vulkan / sum_vulkan) ** 2) ** 0.5

#ratio -= 1
#ratio *= 100
#ratio_std *= 100

#latex_table += f"\nAverage VkFFT (Vulkan) / VkFFT (CUDA) Ratio: & \\multicolumn{{3}}{{c}}{{{ratio:.3f} $\\pm$ {ratio_std:.4f}}} \\\\\n"

with open('1d_fft.tex', 'w') as f:
    f.write(latex_table)
