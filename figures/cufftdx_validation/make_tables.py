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

cufft_data = get_test_data('../../test_results/conv_scaled_nvidia/cufft.csv')
cufftdx_data = get_test_data('../../test_results/conv_scaled_nvidia/cufftdx.csv')

control_latex_table = """\\begin{table}[htbp]
\\caption{Comparison of Execution time (ms) for Batched Scaling Convolution Operations from Nvidia's benchmark}
\\begin{center}
\\begin{tabular}{c c c c}
\\hline
\\textbf{FFT Size ($N$)} & \\textbf{cuFFT} & \\textbf{cuFFTdx} & \\textbf{Ratio} \\\\
\\hline
"""

for size in sorted(cufft_data.keys()):
    cufft_mean, cufft_std = cufft_data[size]
    cufftdx_mean, cufftdx_std = cufftdx_data[size]
    ratio = cufft_mean / cufftdx_mean
    ratio_std = ratio * ((cufftdx_std / cufftdx_mean) ** 2 + (cufft_std / cufft_mean) ** 2) ** 0.5

    control_latex_table += f"{size} & {cufft_mean:.1f} $\\pm$ {cufft_std:.3f} & {cufftdx_mean:.2f} $\\pm$ {cufftdx_std:.4f} & {ratio:.3f} $\\pm$ {ratio_std:.4f} \\\\\n"

control_latex_table += """\\hline
\\end{tabular}
\\label{tab:fft_results}
\\end{center}
\\end{table}"""

with open('nvidia_control.tex', 'w') as f:
    f.write(control_latex_table)

my_cufft_data = get_test_data('../../test_results/conv_scaled_control/cufft.csv')
my_cufftdx_data = get_test_data('../../test_results/conv_scaled_control/zipfft.csv')

latex_table = """\\begin{table}[htbp]
\\caption{Comparison of Effective Bandwidth (GB/s) for Batched Scaling Convolution Operations}
\\begin{center}
\\begin{tabular}{c c c c}
\\hline
\\textbf{FFT Size ($N$)} & \\textbf{cuFFT} & \\textbf{cuFFTdx} & \\textbf{Ratio} \\\\
\\hline
"""

for size in sorted(my_cufft_data.keys()):
    cufft_mean, cufft_std = my_cufft_data[size]
    cufftdx_mean, cufftdx_std = my_cufftdx_data[size]
    ratio = cufftdx_mean / cufft_mean
    ratio_std = ratio * ((cufftdx_std / cufftdx_mean) ** 2 + (cufft_std / cufft_mean) ** 2) ** 0.5

    latex_table += f"{size} & {cufft_mean:.1f} $\\pm$ {cufft_std:.3f} & {cufftdx_mean:.2f} $\\pm$ {cufftdx_std:.4f} & {ratio:.3f} $\\pm$ {ratio_std:.4f} \\\\\n"

latex_table += """\\hline
\\end{tabular}
\\label{tab:my_fft_results}
\\end{center}
\\end{table}"""

with open('custom_cuda.tex', 'w') as f:
    f.write(latex_table)

my_vkdispatch_data = get_test_data('../../test_results/conv_scaled_control/vkdispatch.csv')
my_vkdispatch_naive_data = get_test_data('../../test_results/conv_scaled_control/vkdispatch_naive.csv')

latex_table = """\\begin{table}[htbp]
\\caption{Comparison of Effective Bandwidth (GB/s) for Batched Scaling Convolution Operations in vkdispatch}
\\begin{center}
\\begin{tabular}{c c c c}
\\hline
\\textbf{FFT Size ($N$)} & \\textbf{vkdispatch (naive)} & \\textbf{vkdispatch (fused)} & \\textbf{Ratio} \\\\
\\hline
"""

for size in sorted(my_vkdispatch_data.keys()):
    vkdispatch_mean, vkdispatch_std = my_vkdispatch_data[size]
    vkdispatch_naive_mean, vkdispatch_naive_std = my_vkdispatch_naive_data[size]
    ratio = vkdispatch_mean / vkdispatch_naive_mean
    ratio_std = ratio * ((vkdispatch_naive_std / vkdispatch_naive_mean) ** 2 + (vkdispatch_std / vkdispatch_mean) ** 2) ** 0.5

    latex_table += f"{size} & {vkdispatch_naive_mean:.2f} $\\pm$ {vkdispatch_naive_std:.4f} & {vkdispatch_mean:.1f} $\\pm$ {vkdispatch_std:.3f} & {ratio:.3f} $\\pm$ {ratio_std:.4f} \\\\\n"

latex_table += """\\hline
\\end{tabular}
\\label{tab:my_fft_results}
\\end{center}
\\end{table}"""

with open('custom_vulkan.tex', 'w') as f:
    f.write(latex_table)