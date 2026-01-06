#! /bin/python3

import os
import subprocess
from pathlib import Path
import platform
import signal
import sys
import atexit

import make_graph

DATA_SIZE=2**27
ITER_COUNT=200
BATCH_SIZE=20
REPEATS=3

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

def run_process(command, capture_stdout=False, cwd=None):
    """Run a subprocess and track it for cleanup"""
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE if capture_stdout else None,
        stderr=subprocess.PIPE if capture_stdout else None,
        text=True if capture_stdout else None,
        cwd=cwd,
        start_new_session=True
    )
    child_processes.append(proc)

    return_val = None

    if capture_stdout:
        return_val = proc.communicate()
    else:
        proc.wait()
    child_processes.remove(proc)

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, " ".join(command))

    return return_val

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

__cuda_info = None
cuda_enabled = sys.argv.count('--cuda') > 0

def get_cuda_info():
    global __cuda_info

    if __cuda_info is not None:
        return __cuda_info

    cuda_home_dir = os.environ.get('CUDA_HOME', None)

    nvcc_dir = "nvcc"

    if cuda_home_dir is not None:
        nvcc_dir = os.path.join(cuda_home_dir, "bin", "nvcc")

    # Run the program and capture its stdout as a string
    run_process([nvcc_dir, 'arch_code.cu', '-o', 'arch_code.exec'])

    # Run the program and capture its stdout as a string
    result = run_process(
        ['./arch_code.exec'],
        capture_stdout=True
    )

    __cuda_info = (nvcc_dir, int(result[0].strip()))

    return __cuda_info

def cufftdx_test(test_name: str, nvcc_dir: str, cuda_arch: int):
    if platform.system() == "Darwin":
        print(f"Skipping {test_name} cuFFTdx test on macOS")
        return

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
                        str(2)],
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
                        str(0)],
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
    
    make_graph.make_graph(test_name, title, xlabel, ylabel)

def run_test(test_name: str, title: str, xlabel: str, ylabel: str):
    print(f"Running {test_name} test...")
    
    if not os.path.isdir(f"tests/{test_name}/test_results"):
        os.mkdir(f"tests/{test_name}/test_results")

    if cuda_enabled or sys.argv.count('--validate') > 0:
        nvcc_dir, cuda_arch = get_cuda_info()
        cufftdx_test(test_name, nvcc_dir, cuda_arch)

    if sys.argv.count('--validate') > 0:
        return

    print(f"Running VkDispatch {test_name} test...")
    run_process(['python3', '../vkdispatch_test.py',
             str(DATA_SIZE),
             str(ITER_COUNT),
             str(BATCH_SIZE),
             str(REPEATS)],
            cwd=Path(f"tests/{test_name}/test_results").resolve())
    
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

    make_graph.make_graph(test_name, title, xlabel, ylabel)

if __name__ == "__main__":
    fetch_dependencies()

    # run_test(
    #     test_name="fft_nonstrided",
    #     title="Nonstrided FFT Performance",
    #     xlabel="FFT Size",
    #     ylabel="GB/s (higher is better)"
    # )

    # run_test(
    #     test_name="fft_strided",
    #     title="Strided FFT Performance",
    #     xlabel="FFT Size",
    #     ylabel="GB/s (higher is better)"
    # )

    # run_test(
    #     test_name="fft_2d",
    #     title="2D FFT Performance",
    #     xlabel="FFT Size",
    #     ylabel="GB/s (higher is better)"
    # )

    # if cuda_enabled:
    #     run_nvidia_test(
    #         test_name="conv_scaled_nvidia",
    #         title="NVidia Scaled Convolution Performance",
    #         xlabel="Convolution Size (FFT size)",
    #         ylabel="ms (lower is better)"
    #     )

    # run_test(
    #     test_name="conv_scaled_control",
    #     title="Control Scaled Convolution Performance",
    #     xlabel="Convolution Size (FFT size)", 
    #     ylabel="GB/s (higher is better)"
    # )

    run_test(
        test_name="conv_2d",
        title="2D Convolution Performance",
        xlabel="Convolution Size (FFT size)", 
        ylabel="GB/s (higher is better)"
    )

    # run_test(
    #     test_name="conv_2d_padded",
    #     title="2D Padded Convolution Performance",
    #     xlabel="Convolution Size (FFT size)", 
    #     ylabel="GB/s (higher is better)"
    # )