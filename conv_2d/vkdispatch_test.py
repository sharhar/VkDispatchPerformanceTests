import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import entrypoint, run_vkdispatch, Config

import vkdispatch as vd
import vkdispatch.codegen as vc

def test_function(config: Config,
                    fft_size: int,
                    buffer: vd.Buffer,
                    kernel: vd.Buffer):
    assert kernel.shape == buffer.shape

    vd.fft.convolve2D(buffer, kernel)

def test_function_transpose(config: Config,
                    fft_size: int,
                    buffer: vd.Buffer,
                    kernel: vd.Buffer):
    assert kernel.size >= vd.fft.get_transposed_size(buffer.shape, axis=1)
    vd.fft.convolve2D(buffer, kernel, transposed_kernel=True)

if __name__ == "__main__":
    entrypoint("vkdispatch", run_vkdispatch, 11, test_function)
    entrypoint("vkdispatch_transpose", run_vkdispatch, 11, test_function_transpose)