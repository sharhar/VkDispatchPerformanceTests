import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import run_test, Config

import numpy as np
import vkdispatch as vd
import vkdispatch.codegen as vc

@vd.shader("buff.size")
def scaling_shader(buff: vc.Buffer[vc.c64], scale_factor: vc.Var[vc.f32]):
    idx = vc.global_invocation_id().x
    buff[idx] = buff[idx] * scale_factor

def test_function_naive(config: Config,
                    fft_size: int,
                    buffer: vd.Buffer,
                    kernel: vd.Buffer):
    
    vd.vkfft.fft(buffer)
    scaling_shader(buffer, np.random.rand())
    vd.vkfft.ifft(buffer)

if __name__ == "__main__":
    run_test("vkfft", 6, test_function_naive)