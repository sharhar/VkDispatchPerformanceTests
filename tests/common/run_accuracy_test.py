import vkdispatch as vd

from typing import Callable, Tuple

import dataclasses
import csv
import sys
import numpy as np


@dataclasses.dataclass
class AccuracyConfig:
    data_size: int
    iter_count: int
    iter_batch: int
    run_count: int
    seed: int = 1337

    def make_shape(self, fft_size: int) -> Tuple[int, ...]:
        total_square_size = fft_size * fft_size
        assert self.data_size % total_square_size == 0, "Data size must be a multiple of fft_size squared"
        return (self.data_size // total_square_size, fft_size, fft_size)

    def make_random_data(self, fft_size: int, run_index: int):
        shape = self.make_shape(fft_size)
        rng = np.random.default_rng(self.seed + fft_size * 1000 + run_index)

        real = rng.standard_normal(shape).astype(np.float32)
        imag = rng.standard_normal(shape).astype(np.float32)
        return (real + 1j * imag).astype(np.complex64)


def parse_args() -> AccuracyConfig:
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <data_size> <iter_count> <iter_batch> <run_count>")
        sys.exit(1)

    config = AccuracyConfig(
        data_size=int(sys.argv[1]),
        iter_count=int(sys.argv[2]),
        iter_batch=int(sys.argv[3]),
        run_count=int(sys.argv[4]),
    )

    if config.run_count <= 0:
        raise ValueError("run_count must be positive")

    return config


def get_fft_sizes():
    return [2**i for i in range(3, 13)]  # FFT sizes from 8 to 4096 (inclusive)


def _to_buffer_layout(data: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(data.astype(np.complex64, copy=False)) #.view(np.float32)


def _from_buffer_layout(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.complexfloating):
        return np.ascontiguousarray(data.astype(np.complex64, copy=False))

    if data.shape[-1] != 2:
        raise ValueError(f"Expected trailing dimension of 2 for complex view, got shape {data.shape}")

    float_data = np.ascontiguousarray(data.astype(np.float32, copy=False))
    return float_data.view(np.complex64).reshape(float_data.shape[:-1])


def _compute_metrics(reference: np.ndarray, result: np.ndarray):
    reference64 = reference.astype(np.complex128, copy=False)
    result64 = result.astype(np.complex128, copy=False)

    delta = result64 - reference64
    abs_delta = np.abs(delta)
    abs_reference = np.abs(reference64)

    eps = 1e-12
    relative_l2 = np.linalg.norm(delta.ravel()) / max(np.linalg.norm(reference64.ravel()), eps)
    max_relative = np.max(abs_delta / np.maximum(abs_reference, eps))
    max_absolute = np.max(abs_delta)

    return float(relative_l2), float(max_relative), float(max_absolute)


def run_backend_once(
        config: AccuracyConfig,
        fft_size: int,
        run_index: int,
        gpu_function: Callable):
    input_data = config.make_random_data(fft_size, run_index)
    reference = np.fft.fft(input_data)

    shape = config.make_shape(fft_size)

    buffer = vd.Buffer(shape, var_type=vd.complex64)
    kernel = vd.Buffer(shape, var_type=vd.complex64)

    try:
        buffer.write(input_data) #_to_buffer_layout(input_data))
        gpu_function(config, fft_size, buffer, kernel)
        vd.queue_wait_idle()

        result_data = buffer.read(0) #_from_buffer_layout(buffer.read(0))
    finally:
        buffer.destroy()
        kernel.destroy()
        vd.queue_wait_idle()

    vd.fft.cache_clear()

    return _compute_metrics(reference, result_data)


def run_accuracy_test(output_name: str,
                      backend_name: str,
                      gpu_function: Callable):
    config = parse_args()
    fft_sizes = get_fft_sizes()

    with open(f"{output_name}.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ['Backend', 'FFT Size']
            + [f'Run {i + 1} (Relative L2 Error)' for i in range(config.run_count)]
            + ['Mean', 'Std Dev', 'Worst Max Relative Error', 'Worst Max Absolute Error']
        )

        for fft_size in fft_sizes:
            rel_l2_errors = []
            max_relative_errors = []
            max_absolute_errors = []

            for run_index in range(config.run_count):
                relative_l2, max_relative, max_absolute = run_backend_once(
                    config, fft_size, run_index, gpu_function)
                rel_l2_errors.append(relative_l2)
                max_relative_errors.append(max_relative)
                max_absolute_errors.append(max_absolute)

                print(
                    f"FFT Size: {fft_size}, "
                    f"Relative L2 Error: {relative_l2:.6e}, "
                    f"Max Relative Error: {max_relative:.6e}, "
                    f"Max Absolute Error: {max_absolute:.6e}"
                )

            writer.writerow(
                [backend_name, fft_size]
                + [f"{value:.6e}" for value in rel_l2_errors]
                + [
                    f"{np.mean(rel_l2_errors):.6e}",
                    f"{np.std(rel_l2_errors):.6e}",
                    f"{np.max(max_relative_errors):.6e}",
                    f"{np.max(max_absolute_errors):.6e}",
                ]
            )

    print(f"Accuracy results saved to {output_name}.csv")
