import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import entrypoint, run_torch, Config

import torch

def test_function(config: Config,
                    fft_size: int,
                    buffer: torch.Tensor,
                    kernel: torch.Tensor) -> torch.Tensor:
    signal_size = fft_size // config.signal_factor
    torch.fft.ifft2(torch.fft.fft2(buffer, s=(signal_size, signal_size))  * kernel)

if __name__ == "__main__":
    entrypoint("torch", run_torch, 11, test_function)