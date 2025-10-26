import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import entrypoint, run_torch, Config

import torch
from zipfft import fft_strided

def test_function(config: Config,
                    fft_size: int,
                    buffer: torch.Tensor,
                    kernel: torch.Tensor) -> torch.Tensor:
    fft_strided.fft(buffer, False)

def test_function_smem(config: Config,
                    fft_size: int,
                    buffer: torch.Tensor,
                    kernel: torch.Tensor) -> torch.Tensor:
    fft_strided.fft(buffer, True)

if __name__ == "__main__":
    entrypoint("zipfft", run_torch, 2, test_function)
    entrypoint("zipfft_smem", run_torch, 2, test_function_smem)