import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import entrypoint, run_vkdispatch, Config

import vkdispatch as vd
import vkdispatch.codegen as vc

@vd.shader("buffer.size")
def add_shader(buffer: vc.Buff[vc.c64]):
    idx = vc.global_invocation_id().x
    buffer[idx] += 1

def test_function(config: Config,
                    fft_size: int,
                    buffer: vd.Buffer,
                    kernel: vd.Buffer):
    add_shader(buffer)
    #vd.fft.fft(buffer)

if __name__ == "__main__":
    entrypoint("vkdispatch", run_vkdispatch, 2, test_function)