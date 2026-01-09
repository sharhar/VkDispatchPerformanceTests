import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import run_test, Config

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

    # Transpose io index to avoid excluded regions in the padded buffer
    def transpose_index(io_index: vc.ShaderVariable):
        return vc.unravel_index(
            vc.ravel_index(
                io_index,
                trimmed_shape
            ).to_register(),
            buffer.shape
        )

    @vd.map 
    def input_mapping(input_buffer: vc.Buffer[vc.c64]):
        read_op = vd.fft.read_op()

        read_op.read_from_buffer(
            input_buffer,
            io_index=transpose_index(read_op.io_index)
        )

    @vd.map
    def output_mapping(output_buffer: vc.Buffer[vc.c64]):
        write_op = vd.fft.write_op()

        write_op.write_to_buffer(
            output_buffer,
            io_index=transpose_index(write_op.io_index)
        )

    # Do the first FFT on the correlation buffer accross the first axis
    vd.fft.fft(
        buffer,
        buffer,
        buffer_shape=(
            buffer.shape[0],
            signal_shape[1],
            buffer.shape[2]
        ),
        input_map=input_mapping,
        output_map=output_mapping,
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

@vd.shader("buff.size")
def convolve_naive(buff: vc.Buff[vc.c64], kernel: vc.Buff[vc.c64]):
    ind = vc.global_invocation_id().x
    buff[ind] = vc.mult_complex(buff[ind], kernel[ind].conjugate())

def test_function_naive(config: Config,
                    fft_size: int,
                    buffer: vd.Buffer,
                    kernel: vd.Buffer):
    vd.fft.fft2(buffer)
    convolve_naive(buffer, kernel)
    vd.fft.ifft2(buffer)

if __name__ == "__main__":
    run_test("vkdispatch", 11, test_function)
    run_test("vkdispatch_transpose", 11, test_function_transpose)
    run_test("vkdispatch_naive", 11, test_function_naive)