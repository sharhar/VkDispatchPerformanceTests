#! /bin/python3

import os
import subprocess
from pathlib import Path
import signal
import sys
import atexit

import glob
import csv
from typing import Dict, Tuple, Set, List

import numpy as np


# Nested structure:
# merged[backend][fft_size] = (mean, std)
MergedType = Dict[str, Dict[int, Tuple[float, float]]]


from matplotlib import pyplot as plt
import shutil
import matplotlib.colors as mcolors
import colorsys

DATA_SIZE=2**27
ITER_COUNT=200
BATCH_SIZE=20
REPEATS=3

__cuda_info = None
cuda_enabled = sys.argv.count('--cuda') > 0
opencl_enabled = sys.argv.count('--opencl') > 0
vulkan_enabled = sys.argv.count('--vulkan') > 0


child_processes = []

def cleanup_children(signum=None, frame=None):
    for proc in child_processes:
        try:
            if proc.poll() is None:
                proc.kill()
        except:
            pass
    child_processes.clear()
    if signum is not None:
        sys.exit(1)

atexit.register(cleanup_children)
signal.signal(signal.SIGINT, cleanup_children)
signal.signal(signal.SIGTERM, cleanup_children)

def adjust_lightness(color, factor):
    """Lighten or darken a given matplotlib color by multiplying its lightness by 'factor'."""
    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color
    r, g, b = mcolors.to_rgb(c)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, min(1, l * factor))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)

color0 = plt.cm.tab10(0)  # Blue
color1 = plt.cm.tab10(1)  # Orange
color2 = plt.cm.tab10(2)  # Green
color3 = plt.cm.tab10(3)  # Red
color4 = plt.cm.tab10(4)  # Purple
color5 = plt.cm.tab10(5)  # Brown
color6 = plt.cm.tab10(6)  # Pink
color7 = plt.cm.tab10(7)  # Gray
color8 = plt.cm.tab10(8)  # Olive
color9 = plt.cm.tab10(9)  # Cyan

backend_list = [
    "cufftdx",
    "cufftdx_naive",
    "cufft",

    "cufftdx_nvidia",
    "cufft_nvidia",

    "vkdispatch_vulkan",
    "vkdispatch_naive_vulkan",
    "vkdispatch_accuracy_vulkan",

    "vkdispatch_cuda",
    "vkdispatch_naive_cuda",
    "vkdispatch_accuracy_cuda",

    "vkdispatch_opencl",
    "vkdispatch_naive_opencl",
    "vkdispatch_accuracy_opencl",

    "vkfft_vulkan",
    "vkfft_naive_vulkan",
    "vkfft_accuracy_vulkan",
]

color_dict = {
    "cufftdx": adjust_lightness(color2, 1.4),
    "cufftdx_naive": adjust_lightness(color2, 1.2),
    "cufft": adjust_lightness(color2, 0.8),

    "cufftdx_nvidia": adjust_lightness(color6, 1.2),
    "cufft_nvidia": adjust_lightness(color6, 0.8),

    "vkdispatch_vulkan": adjust_lightness(color3, 1.2),
    "vkdispatch_naive_vulkan": adjust_lightness(color3, 0.8),
    "vkdispatch_accuracy_vulkan": adjust_lightness(color3, 0.8),

    "vkdispatch_cuda": adjust_lightness(color8, 1.2),
    "vkdispatch_naive_cuda": adjust_lightness(color8, 0.8),
    "vkdispatch_accuracy_cuda": adjust_lightness(color8, 0.8),

    "vkdispatch_opencl": adjust_lightness(color0, 1.2),
    "vkdispatch_naive_opencl": adjust_lightness(color0, 0.8),
    "vkdispatch_accuracy_opencl": adjust_lightness(color0, 0.8),

    "vkfft_vulkan": adjust_lightness(color9, 1.2),
    "vkfft_naive_vulkan": adjust_lightness(color9, 0.8),
    "vkfft_accuracy_vulkan": adjust_lightness(color9, 0.8),
}

def get_backend_color(backend_name: str) -> Tuple[float, float, float]:
    return color_dict.get(backend_name, (0.5, 0.5, 0.5))  # Default to gray if unknown


def sort_backend(backends: Set[str]) -> List[str]:
    sorted_list = []
    
    for backend in backend_list:
        if backend in backends:
            sorted_list.append(backend)

    return sorted_list

def read_bench_csvs(pattern) -> Tuple[MergedType, Set[str], Set[int]]:
    files = glob.glob(pattern)

    merged: MergedType = {}
    backends: Set[str] = set()
    fft_sizes: Set[int] = set()

    for filename in files:
        print(f'Reading: {filename}')

        with open(filename, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                backend = row['Backend'].strip()
                size = int(row['FFT Size'])
                mean = float(row['Mean'])
                std = float(row['Std Dev'])

                backends.add(backend)
                fft_sizes.add(size)

                if backend not in merged:
                    merged[backend] = {}

                # last one wins if duplicates appear across files
                merged[backend][size] = (mean, std)

    return merged, backends, fft_sizes

def save_line_graph(
        backends: Set[str],
        fft_sizes: Set[int],
        merged: MergedType,
        outfile: str,
        title: str,
        xlabel: str,
        ylabel: str):
    plt.figure(figsize=(10, 6))

    used_fft_sizes = sorted(fft_sizes)

    for backend_name in sort_backend(backends):
        means = [
            merged[backend_name][i][0]
            for i in used_fft_sizes
        ]
        stds = [
            merged[backend_name][i][1]
            for i in used_fft_sizes
        ]
        
        plt.errorbar(
            used_fft_sizes,
            means,
            yerr=stds,
            label=backend_name,
            capsize=5,
            color=get_backend_color(backend_name)
        )
    plt.xscale('log', base=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(outfile)

def copy_files(src_dir: str, dst_dir: str, extensions: List[str] = None):
    """Copy files with specified extensions from src_dir to dst_dir."""
    
    if extensions is None:
        extensions = ['.csv', '.png']
    
    os.makedirs(dst_dir, exist_ok=True)
    
    for ext in extensions:
        pattern = os.path.join(src_dir, f'*{ext}')
        files = glob.glob(pattern)
        for src_file in files:
            dst_file = os.path.join(dst_dir, os.path.basename(src_file))
            shutil.copy2(src_file, dst_file)
            print(f'Copied: {src_file} -> {dst_file}')

def make_graph(test_name: str, title: str, xlabel: str, ylabel: str):
    src_dir = os.path.join("tests", test_name, "test_results")
    out_dir = os.path.join("test_results", test_name)

    copy_files(src_dir, out_dir)

    merged, backends, fft_sizes = read_bench_csvs(os.path.join(out_dir, "*.csv"))

    print('\nSummary:')
    print(f'Backends found: {sorted(backends)}')
    print(f'Convolution sizes found: {sorted(fft_sizes)}')
    print(f'Total entries: {sum(len(v) for v in merged.values())}')

    sorted_backends = sorted(backends)
    sorted_fft_sizes = sorted(fft_sizes)

    graph_filename = os.path.join(out_dir, "graph.png")
    save_line_graph(
        sorted_backends,
        sorted_fft_sizes,
        merged,
        graph_filename,
        title,
        xlabel,
        ylabel
    )

    dst_graph = os.path.join("test_results", f"{test_name}_graph.png")

    shutil.copy2(graph_filename, dst_graph)
    print(f'Copied: {graph_filename} -> {dst_graph}')

def run_process(command, capture_stdout=False, cwd=None, env=None):
    """Run a subprocess and track it for cleanup.

    Args:
        command: Command list to execute.
        capture_stdout: Whether to capture stdout/stderr.
        cwd: Working directory for the subprocess.
        env: Optional dict of environment variables to add/override.
    """
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE if capture_stdout else None,
        stderr=subprocess.PIPE if capture_stdout else None,
        text=capture_stdout,
        cwd=cwd,
        env=proc_env,
        start_new_session=True,
    )
    child_processes.append(proc)

    try:
        if capture_stdout:
            return_val = proc.communicate()
        else:
            proc.wait()
            return_val = None

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, command)

        return return_val
    finally:
        if proc in child_processes:
            child_processes.remove(proc)

def fetch_dependencies():
    if os.path.isdir('dependencies'):
        print("Dependencies already fetched")
        return

    os.mkdir('dependencies')
    os.chdir('dependencies')
    run_process([
        'wget',
        'https://developer.nvidia.com/downloads/compute/cuFFTDx/redist/cuFFTDx/cuda12/nvidia-mathdx-25.06.1-cuda12.tar.gz'
    ])
    run_process([
        'tar',
        '-xvf',
        'nvidia-mathdx-25.06.1-cuda12.tar.gz'
    ])
    run_process([
        'rm',
        'nvidia-mathdx-25.06.1-cuda12.tar.gz'
    ])

    run_process([
        'git',
        'clone',
        'https://github.com/NVIDIA/cutlass.git'
    ])
    os.chdir('cutlass')
    run_process([
        'git',
        'checkout',
        'e6e2cc29f5e7611dfc6af0ed6409209df0068cf2'
    ])
    os.chdir('..')
    os.chdir('..')

def get_cuda_info():
    global __cuda_info

    if __cuda_info is not None:
        return __cuda_info

    cuda_home_dir = os.environ.get('CUDA_HOME', None)

    nvcc_dir = "nvcc"

    if cuda_home_dir is not None:
        nvcc_dir = os.path.join(cuda_home_dir, "bin", "nvcc")

    arch_code_program = """
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
        std::fprintf(stderr, "No CUDA devices found: %s\\n", cudaGetErrorString(e));
        return 1;
    }
    if (dev < 0 || dev >= count) {
        std::fprintf(stderr, "Invalid device index %d (0..%d)\\n", dev, count - 1);
        return 1;
    }

    cudaDeviceProp prop{};
    e = cudaGetDeviceProperties(&prop, dev);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceProperties failed: %s\\n", cudaGetErrorString(e));
        return 1;
    }

    std::printf("%d%d\\n", prop.major, prop.minor); // e.g., 86
    return 0;
}
"""

    with open("arch_code.cu", "w") as f:
        f.write(arch_code_program)

    # Run the program and capture its stdout as a string
    run_process([nvcc_dir, 'arch_code.cu', '-o', 'arch_code.exec', "-Wno-deprecated-gpu-targets"])

    # Run the program and capture its stdout as a string
    result = run_process(
        ['./arch_code.exec'],
        capture_stdout=True
    )

    # Delete the executable and source file
    os.remove('arch_code.cu')
    os.remove('arch_code.exec')

    __cuda_info = (nvcc_dir, int(result[0].strip()))

    return __cuda_info

def cufftdx_test(test_name: str, nvcc_dir: str, cuda_arch: int):
    if not os.path.isfile(f"tests/{test_name}/cufftdx_test.cu"):
        print(f"Skipping {test_name} cuFFTdx test - cufft_test.cu not found")
        return
    
    print(f"Compiling {test_name} cuFFTdx test...")
    run_process([nvcc_dir,
                 "../cufftdx_test.cu",
                 "-std=c++17", "-O3",
                 "-I ../../../dependencies/cutlass/include",
                 "-I ../../../dependencies/nvidia-mathdx-25.06.1/nvidia/mathdx/25.06/include",
                 "-DFFTS_PER_BLOCK=4",
                 f"-DARCH={cuda_arch}0",
                 "-gencode", f"arch=compute_{cuda_arch},code=sm_{cuda_arch}",
                 "-lcufft", "-lculibos",
                 "-o",  "cufftdx_test.exec"],
                 cwd=Path(f"tests/{test_name}/test_results").resolve())
    
    if sys.argv.count('--validate') > 0:
        print(f"Running {test_name} cuFFTdx validation test...")
        run_process([f"./cufftdx_test.exec",
                        str(DATA_SIZE),
                        str(ITER_COUNT),
                        str(BATCH_SIZE),
                        str(REPEATS),
                        str(0)],
                        cwd=Path(f"tests/{test_name}/test_results").resolve())
    else:
        print(f"Running {test_name} cuFFT test...")
        run_process([f"./cufftdx_test.exec",
                        str(DATA_SIZE),
                        str(ITER_COUNT),
                        str(BATCH_SIZE),
                        str(REPEATS),
                        str(1)],
                        cwd=Path(f"tests/{test_name}/test_results").resolve())
        
        print(f"Running {test_name} cuFFTdx test...")
        run_process([f"./cufftdx_test.exec",
                        str(DATA_SIZE),
                        str(ITER_COUNT),
                        str(BATCH_SIZE),
                        str(REPEATS),
                        str(2)],
                        cwd=Path(f"tests/{test_name}/test_results").resolve())
        print(f"Running {test_name} cuFFTdx naive test...")
        run_process([f"./cufftdx_test.exec",
                        str(DATA_SIZE),
                        str(ITER_COUNT),
                        str(BATCH_SIZE),
                        str(REPEATS),
                        str(3)],
                        cwd=Path(f"tests/{test_name}/test_results").resolve())
    
    os.remove(f"tests/{test_name}/test_results/cufftdx_test.exec")

def run_nvidia_test(test_name: str, title: str, xlabel: str, ylabel: str):
    print(f"Running {test_name} custom test script...")

    _, cuda_arch = get_cuda_info()
    run_process(['bash', 'run_test.sh',
            str(DATA_SIZE),
            str(ITER_COUNT),
            str(BATCH_SIZE),
            str(REPEATS),
            str(cuda_arch)],
            cwd=Path(f"tests/{test_name}").resolve())
    
    make_graph(test_name, title, xlabel, ylabel)

def run_test(test_name: str, title: str, xlabel: str, ylabel: str):
    print(f"Running {test_name} test...")

    if not os.path.isdir(f"tests/{test_name}/test_results"):
        os.mkdir(f"tests/{test_name}/test_results")
        
    if cuda_enabled or sys.argv.count('--validate') > 0:
        nvcc_dir, cuda_arch = get_cuda_info()
        cufftdx_test(test_name, nvcc_dir, cuda_arch)

    if sys.argv.count('--validate') > 0:
        return

    if vulkan_enabled:
        if os.path.isfile(f"tests/{test_name}/vkfft_test.py"):
            print(f"Running VKFFT {test_name} test...")
            run_process(['python3', '../vkfft_test.py',
                    str(DATA_SIZE),
                    str(ITER_COUNT),
                    str(BATCH_SIZE),
                    str(REPEATS)],
                    cwd=Path(f"tests/{test_name}/test_results").resolve())
        else:
            print(f"Skipping {test_name} VKFFT test - vkfft_test.py not found")
        
        print(f"Running VkDispatch Vulkan {test_name} test...")
        run_process(['python3', '../vkdispatch_test.py',
                str(DATA_SIZE),
                str(ITER_COUNT),
                str(BATCH_SIZE),
                str(REPEATS)],
                cwd=Path(f"tests/{test_name}/test_results").resolve())
    
    if cuda_enabled:
        print(f"Running VkDispatch CUDA {test_name} test...")
        run_process(['python3', '../vkdispatch_test.py',
                str(DATA_SIZE),
                str(ITER_COUNT),
                str(BATCH_SIZE),
                str(REPEATS)],
                cwd=Path(f"tests/{test_name}/test_results").resolve(),
                env={"VKDISPATCH_BACKEND": "cuda"})
    
    if opencl_enabled:
        print(f"Running VkDispatch OpenCL {test_name} test...")
        run_process(['python3', '../vkdispatch_test.py',
                str(DATA_SIZE),
                str(ITER_COUNT),
                str(BATCH_SIZE),
                str(REPEATS)],
                cwd=Path(f"tests/{test_name}/test_results").resolve(),
                env={"VKDISPATCH_BACKEND": "opencl"})

    make_graph(test_name, title, xlabel, ylabel)

def run_accuraccy_test():
    if not os.path.isdir(f"tests/accuracy/test_results"):
        os.mkdir(f"tests/accuracy/test_results")

    accuracy_data_size = DATA_SIZE // 4

    if vulkan_enabled:
        print(f"Running vulkan Accuracy test...")
        run_process(['python3', '../accuracy_test.py',
                    str(accuracy_data_size),
                    str(ITER_COUNT),
                    str(BATCH_SIZE),
                    str(REPEATS),
                    "--vulkan"],
                cwd=Path(f"tests/accuracy/test_results").resolve(),
                env={"VKDISPATCH_BACKEND": "vulkan"})
    
    if cuda_enabled:
        print(f"Running cuda Accuracy test...")
        run_process(['python3', '../accuracy_test.py',
                    str(accuracy_data_size),
                    str(ITER_COUNT),
                    str(BATCH_SIZE),
                    str(REPEATS),
                    "--cuda"],
                cwd=Path(f"tests/accuracy/test_results").resolve(),
                env={
                    "VKDISPATCH_BACKEND": "cuda",
                    "VKDISPATCH_TEST_NVCC_PATH": get_cuda_info()[0],
                    "VKDISPATCH_TEST_CUDA_ARCH": str(get_cuda_info()[1])
                })
    
    if opencl_enabled:
        print(f"Running opencl Accuracy test...")
        run_process(['python3', '../accuracy_test.py',
                    str(accuracy_data_size),
                    str(ITER_COUNT),
                    str(BATCH_SIZE),
                    str(REPEATS),
                    "--opencl"],
                cwd=Path(f"tests/accuracy/test_results").resolve(),
                env={"VKDISPATCH_BACKEND": "opencl"})
    
    make_graph("accuracy", "Accuracy", "FFT Size", "Error")

if __name__ == "__main__":
    fetch_dependencies()

    run_accuraccy_test()

    run_test(
        test_name="fft_nonstrided",
        title="Nonstrided FFT Performance",
        xlabel="FFT Size",
        ylabel="GB/s (higher is better)"
    )

    run_test(
        test_name="fft_strided",
        title="Strided FFT Performance",
        xlabel="FFT Size",
        ylabel="GB/s (higher is better)"
    )

    run_test(
        test_name="fft_2d",
        title="2D FFT Performance",
        xlabel="FFT Size",
        ylabel="GB/s (higher is better)"
    )

    if cuda_enabled:
        run_nvidia_test(
            test_name="conv_scaled_nvidia",
            title="NVidia Scaled Convolution Performance",
            xlabel="Convolution Size (FFT size)",
            ylabel="ms (lower is better)"
        )

    run_test(
        test_name="conv_scaled_control",
        title="Control Scaled Convolution Performance",
        xlabel="Convolution Size (FFT size)", 
        ylabel="GB/s (higher is better)"
    )

    run_test(
        test_name="conv_2d",
        title="2D Convolution Performance",
        xlabel="Convolution Size (FFT size)", 
        ylabel="GB/s (higher is better)"
    )

    run_test(
        test_name="conv_2d_padded",
        title="2D Padded Convolution Performance",
        xlabel="Convolution Size (FFT size)", 
        ylabel="GB/s (higher is better)"
    )
