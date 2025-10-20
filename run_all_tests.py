#! /bin/python3

import os
import subprocess
from pathlib import Path

import make_graph

DATA_SIZE=2**27
ITER_COUNT=400
BATCH_SIZE=20
REPEATS=5
ARCH=89

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

run_test(
    "fft_nonstrided",
    "Nonstrided FFT Performance",
    "FFT Size",
    "GB/s (higher is better)"
)

# run_test(
#     "convolution_nvidia",
#     "NVidia Scaled Convolution Performance",
#     "Convolution Size (FFT size)",
#     "ms (lower is better)"
# )

# run_test(
#     "convolution_nonstrided_scaled",
#     "Control Scaled Convolution Performance",
#     "Convolution Size (FFT size)", 
#     "s (lower is better)"
# )