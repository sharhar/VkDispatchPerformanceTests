import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import entrypoint, run_torch, Config

import torch

"""

void at::native::elementwise_kernel
void vector_fft

"""

# https://github.com/pytorch/pytorch/issues/42175
# pytorch adds an extra copy for strided operations.
# 2d avoids this because from the pov of pytorch it is nonstrided

def test_function(config: Config,
                    fft_size: int,
                    buffer: torch.Tensor,
                    kernel: torch.Tensor) -> torch.Tensor:
    #print(buffer.shape)
    buffer = torch.fft.fft(buffer, dim=1)

if __name__ == "__main__":
    entrypoint("torch", run_torch, 2, test_function)