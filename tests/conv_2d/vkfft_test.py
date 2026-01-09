import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import run_test, Config

import vkdispatch as vd

def io_count(elem_count: int, fft_size: int) -> int:
    return 10.0 + (fft_size * fft_size) / elem_count

def test_function(config: Config,
                    fft_size: int,
                    buffer: vd.Buffer,
                    kernel: vd.Buffer):
    vd.vkfft.convolve2D(buffer, kernel)

if __name__ == "__main__":
    pass
    #run_test("vkfft", 11, test_function)

