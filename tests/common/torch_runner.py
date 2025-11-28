import torch
import numpy as np
import time

from typing import Callable, Tuple

from .config import Config

def run_torch(config: Config,
              fft_size: int,
              io_count: int,
              gpu_function: Callable):
    shape = config.make_shape(fft_size)
    #random_data = config.make_random_data(fft_size)
    #random_data_kernel = config.make_random_data(fft_size)

    buffer = torch.empty(
        shape,
        dtype=torch.complex64,
        device='cuda'
    )

    kernel = torch.empty(
        shape,
        dtype=torch.complex64,
        device='cuda'
    )

    #buffer.copy_(torch.from_numpy(random_data).to('cuda'))
    #kernel.copy_(torch.from_numpy(random_data_kernel).to('cuda'))

    stream = torch.cuda.Stream()

    torch.cuda.synchronize()
    
    with torch.cuda.stream(stream):
        for _ in range(config.warmup):
            gpu_function(config, fft_size, buffer, kernel)

    torch.cuda.synchronize()

    gb_byte_count = io_count * np.prod(shape) * 8 / (1024 * 1024 * 1024)
    
    g = torch.cuda.CUDAGraph()

    # We capture either 1 or K FFTs back-to-back. All on the same stream.
    with torch.cuda.graph(g, stream=stream):
        for _ in range(max(1, config.iter_batch)):
            gpu_function(config, fft_size, buffer, kernel)   # creates a tensor once during capture

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.cuda.stream(stream):
        for _ in range(config.iter_count // max(1, config.iter_batch)):
            g.replay()

    torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - start_time

    return gb_byte_count, elapsed_time