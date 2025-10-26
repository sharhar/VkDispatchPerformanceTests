import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import entrypoint, run_torch, Config

import torch
from zipfft import fft_nonstrided
from zipfft import conv_strided_padded
from zipfft import fft_nonstrided_padded

def test_function(config: Config,
                    fft_size: int,
                    buffer: torch.Tensor,
                    kernel: torch.Tensor) -> torch.Tensor:
    
    assert fft_nonstrided_padded.get_supported_padding_ratio() == config.signal_factor
    assert conv_strided_padded.get_supported_padding_ratio() == config.signal_factor

    fft_nonstrided_padded.fft(buffer)
    conv_strided_padded.conv(buffer, kernel, False, False)
    fft_nonstrided.ifft(buffer)

def test_function_smem(config: Config,
                    fft_size: int,
                    buffer: torch.Tensor,
                    kernel: torch.Tensor) -> torch.Tensor:
    
    assert fft_nonstrided_padded.get_supported_padding_ratio() == config.signal_factor
    assert conv_strided_padded.get_supported_padding_ratio() == config.signal_factor

    fft_nonstrided_padded.fft(buffer)
    conv_strided_padded.conv(buffer, kernel, False, True)
    fft_nonstrided.ifft(buffer)

def test_function_transpose(config: Config,
                    fft_size: int,
                    buffer: torch.Tensor,
                    kernel: torch.Tensor) -> torch.Tensor:
    
    assert fft_nonstrided_padded.get_supported_padding_ratio() == config.signal_factor
    assert conv_strided_padded.get_supported_padding_ratio() == config.signal_factor

    fft_nonstrided_padded.fft(buffer)
    conv_strided_padded.conv(buffer, kernel, True, False)
    fft_nonstrided.ifft(buffer)

def test_function_transpose_smem(config: Config,
                    fft_size: int,
                    buffer: torch.Tensor,
                    kernel: torch.Tensor) -> torch.Tensor:
    
    assert fft_nonstrided_padded.get_supported_padding_ratio() == config.signal_factor
    assert conv_strided_padded.get_supported_padding_ratio() == config.signal_factor

    fft_nonstrided_padded.fft(buffer)
    conv_strided_padded.conv(buffer, kernel, True, True)
    fft_nonstrided.ifft(buffer)

if __name__ == "__main__":
    entrypoint("zipfft", run_torch, 11, test_function)
    entrypoint("zipfft_smem", run_torch, 11, test_function_smem)
    entrypoint("zipfft_transpose", run_torch, 11, test_function_transpose)
    entrypoint("zipfft_transpose_smem", run_torch, 11, test_function_transpose_smem)