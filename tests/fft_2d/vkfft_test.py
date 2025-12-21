import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import run_test, Config

import vkdispatch as vd

def test_function(config: Config,
                    fft_size: int,
                    buffer: vd.Buffer,
                    kernel: vd.Buffer):
    vd.vkfft.fft2(buffer)

if __name__ == "__main__":
    run_test("vkfft", 4, test_function)