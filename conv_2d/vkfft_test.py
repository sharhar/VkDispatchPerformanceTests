import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import entrypoint, run_vkdispatch, Config

import vkdispatch as vd

def test_function(config: Config,
                    fft_size: int,
                    buffer: vd.Buffer,
                    kernel: vd.Buffer):
    print(f"Running 2D convolution with FFT size {buffer.shape}")
    print(f"Buffer shape: {buffer.shape}, Kernel shape: {kernel.shape}")
    vd.vkfft.convolve2D(buffer, kernel)

if __name__ == "__main__":
    entrypoint("vkfft", run_vkdispatch, 11, test_function)