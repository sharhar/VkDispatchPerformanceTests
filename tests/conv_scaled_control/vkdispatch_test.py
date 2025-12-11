import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import entrypoint, run_vkdispatch, Config

import vkdispatch as vd
import vkdispatch.codegen as vc

import numpy as np

@vd.map
def kernel_mapping(scale_factor: vc.Var[vc.f32]):
    read_op = vd.fft.read_op()
    read_op.register[:] = read_op.register * scale_factor

def test_function(config: Config,
                    fft_size: int,
                    buffer: vd.Buffer,
                    kernel: vd.Buffer):
    vd.fft.convolve(buffer, np.random.rand(), kernel_map=kernel_mapping)

@vd.shader("buff.size")
def scaling_shader(buff: vc.Buffer[vc.c64], scale_factor: vc.Var[vc.f32]):
    idx = vc.global_invocation_id().x
    buff[idx] = buff[idx] * scale_factor

def test_function_naive(config: Config,
                    fft_size: int,
                    buffer: vd.Buffer,
                    kernel: vd.Buffer):
    
    vd.fft.fft(buffer)
    scaling_shader(buffer, np.random.rand())
    vd.fft.ifft(buffer)

if __name__ == "__main__":
    entrypoint("vkdispatch", run_vkdispatch, 6, test_function)
    entrypoint("vkdispatch_naive", run_vkdispatch, 6, test_function_naive)