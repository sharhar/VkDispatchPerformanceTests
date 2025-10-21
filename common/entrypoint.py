from .config import Config, get_fft_sizes, parse_args

from typing import Callable

import csv
import numpy as np

def entrypoint(backend_name: str,
               run_function: Callable, 
               io_count: int,
               gpu_function: Callable):
    config = parse_args()
    fft_sizes = get_fft_sizes()

    output_name = f"{backend_name}.csv"
    with open(output_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Backend', 'FFT Size'] + [f'Run {i + 1} (GB/s)' for i in range(config.run_count)] + ['Mean', 'Std Dev'])
        
        for fft_size in fft_sizes:
            rates = []

            for _ in range(config.run_count):
                gb_byte_count, elapsed_time = run_function(config, fft_size, io_count, gpu_function)
                gb_per_second = config.iter_count * gb_byte_count / elapsed_time

                print(f"FFT Size: {fft_size}, Throughput: {gb_per_second:.2f} GB/s")
                rates.append(gb_per_second)

            rounded_data = [round(rate, 2) for rate in rates]
            rounded_mean = round(np.mean(rates), 2)
            rounded_std = round(np.std(rates), 2)

            writer.writerow([backend_name, fft_size] + rounded_data + [rounded_mean, rounded_std])
        
    print(f"Results saved to {output_name}")