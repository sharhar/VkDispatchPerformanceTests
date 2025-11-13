import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import entrypoint, run_torch, Config

import torch
import numpy as np

def test_function(config: Config,
                    fft_size: int,
                    buffer: torch.Tensor,
                    kernel: torch.Tensor) -> torch.Tensor:
    scale_factor = np.random.rand() + 0.5
    torch.fft.ifft(torch.fft.fft(buffer) * scale_factor)

if __name__ == "__main__":
    entrypoint("torch", run_torch, 6, test_function)