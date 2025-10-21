import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import entrypoint, run_vkdispatch, Config

import vkdispatch as vd
import vkdispatch.codegen as vc

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

def test_function(config: Config,
                    fft_size: int,
                    buffer: vd.Buffer,
                    kernel: vd.Buffer):
    vd.fft.convolve2D(buffer, kernel, kernel_map=kernel_mapping)

if __name__ == "__main__":
    entrypoint("vkdispatch", run_vkdispatch, 11, test_function)