import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import entrypoint, run_vkdispatch, Config

import vkdispatch as vd
import vkdispatch.codegen as vc

def padded_cross_correlation(
        buffer: vd.Buffer,
        kernel: vd.Buffer,
        signal_shape: tuple):

    # Fill input buffer with zeros where needed
    @vd.map_registers([vc.c64])
    def initial_input_mapping(input_buffer: vc.Buffer[vc.c64]):
        vc.if_statement(vc.mapping_index() % buffer.shape[2] < signal_shape[1])

        in_layer_index = vc.mapping_index() % (signal_shape[1] * buffer.shape[2])
        out_layer_index = vc.mapping_index() / (signal_shape[1] * buffer.shape[2])
        actual_index = in_layer_index + out_layer_index * (buffer.shape[1] * buffer.shape[2])

        vc.mapping_registers()[0][:] = input_buffer[actual_index]
        vc.else_statement()
        vc.mapping_registers()[0][:] = "vec2(0)"
        vc.end()

    # Remap output indicies to match the actual buffer shape
    @vd.map_registers([vc.c64])
    def initial_output_mapping(output_buffer: vc.Buffer[vc.c64]):
        in_layer_index = vc.mapping_index() % (signal_shape[1] * buffer.shape[2])
        out_layer_index = vc.mapping_index() / (signal_shape[1] * buffer.shape[2])
        actual_index = in_layer_index + out_layer_index * (buffer.shape[1] * buffer.shape[2])
        output_buffer[actual_index] = vc.mapping_registers()[0]

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
    @vd.map_registers([vc.c64])
    def input_mapping(input_buffer: vc.Buffer[vc.c64]):
        in_layer_index = vc.mapping_index() % (
            buffer.shape[1] * buffer.shape[2]
        )

        vc.if_statement(in_layer_index / buffer.shape[2] < signal_shape[1])
        vc.mapping_registers()[0][:] = input_buffer[vc.mapping_index()]
        vc.else_statement()
        vc.mapping_registers()[0][:] = "vec2(0)"
        vc.end()

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
        img_val[:] = vc.mult_conj_c64(read_register, img_val)

    vd.fft.convolve(
        buffer,
        buffer,
        kernel,
        input_map=input_mapping,
        kernel_map=kernel_mapping,
        axis=1
    )

    vd.fft.ifft(buffer)

def test_function(config: Config,
                    fft_size: int,
                    buffer: vd.Buffer,
                    kernel: vd.Buffer):
    signal_size = fft_size // config.signal_factor
    padded_cross_correlation(buffer, kernel, (signal_size, signal_size))

if __name__ == "__main__":
    entrypoint("vkdispatch", run_vkdispatch, 11, test_function)