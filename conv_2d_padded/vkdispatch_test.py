import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import entrypoint, run_vkdispatch, Config

import vkdispatch as vd
import vkdispatch.codegen as vc

def padded_cross_correlation(
        buffer: vd.Buffer,
        kernel: vd.Buffer,
        signal_shape: tuple,
        transposed_kernel: bool):

    trimmed_shape = (
        buffer.shape[0],
        signal_shape[1],
        buffer.shape[2]
    )

    # Fill input buffer with zeros where needed
    @vd.map #_registers([vc.c64])
    def initial_input_mapping(input_buffer: vc.Buffer[vc.c64]):
        read_op = vd.fft.mapped_read_op()
        trimmed_index_vec = vc.ravel_index(read_op.io_index, trimmed_shape).to_register()
        actual_index = vc.unravel_index(trimmed_index_vec, buffer.shape)
        read_op.read_from_buffer(input_buffer, io_index=actual_index)

    # Remap output indicies to match the actual buffer shape
    @vd.map #_registers([vc.c64])
    def initial_output_mapping(output_buffer: vc.Buffer[vc.c64]):
        write_op = vd.fft.mapped_write_op()
        trimmed_index_vec = vc.ravel_index(write_op.io_index, trimmed_shape).to_register()
        actual_index = vc.unravel_index(trimmed_index_vec, buffer.shape)
        write_op.write_to_buffer(output_buffer, io_index=actual_index)

    # Do the first FFT on the correlation buffer accross the first axis
    vd.fft.fft(
        buffer,
        buffer,
        buffer_shape=(
            buffer.shape[0],
            signal_shape[1],
            buffer.shape[2]
        ),
        input_map=initial_input_mapping,
        output_map=initial_output_mapping,
        input_signal_range=(0, signal_shape[1])
    )
    
    vd.fft.convolve(
        buffer,
        kernel,
        transposed_kernel=transposed_kernel,
        axis=1,
        input_signal_range=(0, signal_shape[1])
    )

    vd.fft.ifft(buffer)

def test_function(config: Config,
                    fft_size: int,
                    buffer: vd.Buffer,
                    kernel: vd.Buffer):
    signal_size = fft_size // config.signal_factor
    padded_cross_correlation(buffer, kernel, (signal_size, signal_size), False)

def test_function_transpose(config: Config,
                    fft_size: int,
                    buffer: vd.Buffer,
                    kernel: vd.Buffer):
    signal_size = fft_size // config.signal_factor
    padded_cross_correlation(buffer, kernel, (signal_size, signal_size), True)

if __name__ == "__main__":
    entrypoint("vkdispatch", run_vkdispatch, 11, test_function)
    entrypoint("vkdispatch_transpose", run_vkdispatch, 11, test_function_transpose)