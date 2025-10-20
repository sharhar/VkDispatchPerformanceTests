from .config import Config, get_fft_sizes, parse_args
from .entrypoint import entrypoint

from .torch_runner import run_torch
from .vkdispatch_runner import run_vkdispatch