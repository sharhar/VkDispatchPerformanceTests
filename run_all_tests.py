#! /bin/python3

import os
import subprocess
from pathlib import Path

import make_graph

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

DATA_SIZE=2**27
ITER_COUNT=400
BATCH_SIZE=20
REPEATS=5
ARCH=int(result.stdout.strip())

def run_test(test_name: str, title: str, xlabel: str, ylabel: str):
    print(f"Running {test_name} test...")
    
    subprocess.run(['bash', 'run_test.sh',
             str(DATA_SIZE),
             str(ITER_COUNT),
             str(BATCH_SIZE),
             str(REPEATS),
             str(ARCH)],
            cwd=Path(test_name).resolve())

    make_graph.make_graph(test_name, title, xlabel, ylabel)

# run_test(
#     test_name="fft_nonstrided",
#     title="Nonstrided FFT Performance",
#     xlabel="FFT Size",
#     ylabel="GB/s (higher is better)"
# )

run_test(
    "convolution_nvidia",
    "NVidia Scaled Convolution Performance",
    "Convolution Size (FFT size)",
    "ms (lower is better)"
)

run_test(
    "convolution_nonstrided_scaled",
    "Control Scaled Convolution Performance",
    "Convolution Size (FFT size)", 
    "s (lower is better)"
)