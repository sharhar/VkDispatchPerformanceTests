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

    # Fill input buffer with zeros where needed
    @vd.map #_registers([vc.c64])
    def initial_input_mapping(input_buffer: vc.Buffer[vc.c64]):
        read_op = vd.fft.mapped_read_op()

        vc.if_statement(read_op.io_index % buffer.shape[2] < signal_shape[1])

        in_layer_index = read_op.io_index % (signal_shape[1] * buffer.shape[2])
        out_layer_index = read_op.io_index / (signal_shape[1] * buffer.shape[2])
        actual_index = in_layer_index + out_layer_index * (buffer.shape[1] * buffer.shape[2])

        read_op.register[:] = input_buffer[actual_index]
        vc.else_statement()
        read_op.register[:] = "vec2(0)"
        vc.end()

    # Remap output indicies to match the actual buffer shape
    @vd.map #_registers([vc.c64])
    def initial_output_mapping(output_buffer: vc.Buffer[vc.c64]):
        write_op = vd.fft.mapped_write_op()

        in_layer_index = write_op.io_index % (signal_shape[1] * buffer.shape[2])
        out_layer_index = write_op.io_index / (signal_shape[1] * buffer.shape[2])
        actual_index = in_layer_index + out_layer_index * (buffer.shape[1] * buffer.shape[2])
        output_buffer[actual_index] = write_op.register

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
        output_map=initial_output_mapping
    )

    # Again, we skip reading the zero-padded values from the input
    @vd.map #_registers([vc.c64])
    def input_mapping(input_buffer: vc.Buffer[vc.c64]):
        read_op = vd.fft.mapped_read_op()

        in_layer_index = read_op.io_index % (
            buffer.shape[1] * buffer.shape[2]
        )

        vc.if_statement(in_layer_index / buffer.shape[2] < signal_shape[1])
        read_op.register[:] = input_buffer[read_op.io_index]
        vc.else_statement()
        read_op.register[:] = "vec2(0)"
        vc.end()

    """
    @vd.map_registers([vc.c64])
    def kernel_mapping(kernel_buffer: vc.Buffer[vc.c64]):
        img_val = vc.mapping_registers()[0]
        read_register = vc.mapping_registers()[1]

        # Calculate the invocation within this FFT batch
        in_group_index = vc.local_invocation().y * vc.workgroup_size().x + vc.local_invocation().x
        out_group_index = vc.workgroup().y * vc.num_workgroups().x + vc.workgroup().x
        workgroup_index = in_group_index + out_group_index * (
            vc.workgroup_size().x * vc.workgroup_size().y
        )

        # Calculate the batch index of the FFT
        batch_index = (
            vc.mapping_index()
        ) / (
            vc.workgroup_size().x * vc.workgroup_size().y *
            vc.num_workgroups().x * vc.num_workgroups().y
        )

        # Calculate the transposed index
        transposed_index = workgroup_index + batch_index * (
            vc.workgroup_size().x * vc.workgroup_size().y *
            vc.num_workgroups().x * vc.num_workgroups().y
        )

        read_register[:] = kernel_buffer[transposed_index]
        img_val[:] = vc.mult_conj_c64(read_register, img_val)"""

    vd.fft.convolve(
        buffer,
        buffer,
        kernel,
        input_map=input_mapping,
        transposed_kernel=transposed_kernel,
        #kernel_map=kernel_mapping,
        axis=1
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