import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import entrypoint, run_vkdispatch, Config

import vkdispatch as vd

def test_function(config: Config,
                    fft_size: int,
                    buffer: vd.Buffer,
                    kernel: vd.Buffer):
    vd.vkfft.convolve_2D(buffer, kernel)

if __name__ == "__main__":
    entrypoint("vkfft", run_vkdispatch, 11, test_function)