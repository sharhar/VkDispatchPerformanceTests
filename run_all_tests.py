#! /bin/python3

import os
import subprocess
from pathlib import Path

import platform

import make_graph

DATA_SIZE=2**26
ITER_COUNT=150
BATCH_SIZE=15
REPEATS=3

if platform.system() == "Darwin":
    ARCH=0
else:
    cuda_home_dir = os.environ.get('CUDA_HOME', None)

    nvcc_dir = "nvcc"

    if cuda_home_dir is not None:
        nvcc_dir = os.path.join(cuda_home_dir, "bin", "nvcc")

    # Run the program and capture its stdout as a string
    subprocess.run(
        [nvcc_dir, 'arch_code.cu', '-o', 'arch_code.exec'],
        check=True
    )

    # Run the program and capture its stdout as a string
    result = subprocess.run(
        ['./arch_code.exec'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    ARCH=int(result.stdout.strip())

def run_test(test_name: str, title: str, xlabel: str, ylabel: str):
    print(f"Running {test_name} test...")
    
    subprocess.run(['bash', 'run_test.sh',
             str(DATA_SIZE),
             str(ITER_COUNT),
             str(BATCH_SIZE),
             str(REPEATS),
             str(ARCH)],
            cwd=Path(f"tests/{test_name}").resolve())

    make_graph.make_graph(test_name, title, xlabel, ylabel)

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

if platform.system() != "Darwin":
    run_test(
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