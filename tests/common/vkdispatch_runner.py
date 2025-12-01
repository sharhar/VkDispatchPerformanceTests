import vkdispatch as vd
import numpy as np
import time

#vd.make_context(device_ids=[1])

from typing import Callable, Tuple

from .config import Config

def run_vkdispatch(config: Config,
                    fft_size: int,
                    io_count: int,
                    gpu_function: Callable) -> float:
    shape = config.make_shape(fft_size)
    #random_data = config.make_random_data(fft_size)
    #random_data_kernel = config.make_random_data(fft_size)

    buffer = vd.Buffer(shape, var_type=vd.complex64)
    #buffer.write(random_data)

    kernel = vd.Buffer(shape, var_type=vd.complex64)
    #kernel.write(random_data_kernel)

    graph = vd.CommandGraph()
    old_graph = vd.set_global_graph(graph)
    
    gpu_function(config, fft_size, buffer, kernel)

    vd.set_global_graph(old_graph)

    for _ in range(config.warmup):
        graph.submit(config.iter_batch)

    vd.queue_wait_idle()

    gb_byte_count = io_count * 8 * buffer.size / (1024 * 1024 * 1024)
    
    start_time = time.perf_counter()

    for _ in range(config.iter_count // config.iter_batch):
        graph.submit(config.iter_batch)

    vd.queue_wait_idle()

    elapsed_time = time.perf_counter() - start_time

    buffer.destroy()
    kernel.destroy()
    graph.destroy()
    vd.fft.cache_clear()

    time.sleep(1)

    vd.queue_wait_idle()    

    return gb_byte_count, elapsed_time