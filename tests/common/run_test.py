import vkdispatch as vd

from typing import Callable, Tuple, Union

import dataclasses
import csv
import sys
import time
import numpy as np

@dataclasses.dataclass
class Config:
    data_size: int
    iter_count: int
    iter_batch: int
    run_count: int
    signal_factor: int
    warmup: int = 10

    def make_shape(self, fft_size: int) -> Tuple[int, ...]:
        total_square_size = fft_size * fft_size
        assert self.data_size % total_square_size == 0, "Data size must be a multiple of fft_size squared"
        return (self.data_size // total_square_size, fft_size, fft_size)
    
    def make_random_data(self, fft_size: int):
        shape = self.make_shape(fft_size)
        return np.random.rand(*shape).astype(np.complex64)

def parse_args() -> Config:
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <data_size> <iter_count> <iter_batch> <run_count>")
        sys.exit(1)

    return Config(
        data_size=int(sys.argv[1]),
        iter_count=int(sys.argv[2]),
        iter_batch=int(sys.argv[3]),
        run_count=int(sys.argv[4]),
        signal_factor=8 # Default signal factor
    )

def get_fft_sizes():
    return [2**i for i in range(3, 13)]  # FFT sizes from 64 to 4096 (inclusive)

def run_vkdispatch(config: Config,
                    fft_size: int,
                    io_count: Union[int, Callable],
                    gpu_function: Callable) -> float:
    shape = config.make_shape(fft_size)

    buffer = vd.Buffer(shape, var_type=vd.complex64)
    kernel = vd.Buffer(shape, var_type=vd.complex64)

    graph = vd.CommandGraph()
    old_graph = vd.set_global_graph(graph)
    
    gpu_function(config, fft_size, buffer, kernel)

    vd.set_global_graph(old_graph)

    for _ in range(config.warmup):
        graph.submit(config.iter_batch)

    vd.queue_wait_idle()

    if callable(io_count):
        io_count = io_count(buffer.size, fft_size)

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

def run_test(test_name: str,
               io_count: Union[int, Callable],
               gpu_function: Callable):
    config = parse_args()
    fft_sizes = get_fft_sizes()

    output_name = f"{test_name}.csv"
    with open(output_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Backend', 'FFT Size'] + [f'Run {i + 1} (GB/s)' for i in range(config.run_count)] + ['Mean', 'Std Dev'])
        
        for fft_size in fft_sizes:
            rates = []

            for _ in range(config.run_count):
                gb_byte_count, elapsed_time = run_vkdispatch(config, fft_size, io_count, gpu_function)
                gb_per_second = config.iter_count * gb_byte_count / elapsed_time

                print(f"FFT Size: {fft_size}, Throughput: {gb_per_second:.4f} GB/s")
                rates.append(gb_per_second)

            rounded_data = [round(rate, 4) for rate in rates]
            rounded_mean = round(np.mean(rates), 4)
            rounded_std = round(np.std(rates), 4)

            writer.writerow([test_name, fft_size] + rounded_data + [rounded_mean, rounded_std])
        
    print(f"Results saved to {output_name}")