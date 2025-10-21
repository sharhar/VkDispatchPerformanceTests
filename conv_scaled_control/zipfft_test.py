import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import entrypoint, run_torch, Config

import torch
from zipfft import conv_nonstrided
import numpy as np

def test_function(config: Config,
                    fft_size: int,
                    buffer: torch.Tensor,
                    kernel: torch.Tensor) -> torch.Tensor:
    scale_factor = np.random.rand() + 0.5
    conv_nonstrided.conv(buffer.view(-1, buffer.size(2)), scale_factor)

if __name__ == "__main__":
    entrypoint("zipfft", run_torch, 6, test_function)