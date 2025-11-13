import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import entrypoint, run_vkdispatch, Config

import vkdispatch as vd
import vkdispatch.codegen as vc

import numpy as np

@vd.map
def kernel_mapping(scale_factor: vc.Var[vc.f32]):
    read_op = vd.fft.mapped_read_op()
    read_op.register[:] = read_op.register * scale_factor

def test_function(config: Config,
                    fft_size: int,
                    buffer: vd.Buffer,
                    kernel: vd.Buffer):
    vd.fft.convolve(buffer, np.random.rand(), kernel_map=kernel_mapping)

if __name__ == "__main__":
    entrypoint("vkdispatch", run_vkdispatch, 6, test_function)