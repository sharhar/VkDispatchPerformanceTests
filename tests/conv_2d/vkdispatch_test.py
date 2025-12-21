import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import run_test, Config

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

@vd.shader("buff.size")
def convolve_naive(buff: vc.Buff[vc.c64], kernel: vc.Buff[vc.c64], fft_size: vc.Const[vc.u32]):
    ind = vc.global_invocation_id().x
    buff[ind] = vc.mult_complex(buff[ind], kernel[ind % (fft_size * fft_size)].conjugate())

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