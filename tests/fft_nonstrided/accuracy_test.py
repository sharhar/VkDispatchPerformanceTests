import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import run_accuracy_test, AccuracyConfig

import vkdispatch as vd


def vkdispatch_fft_test_function(config: AccuracyConfig,
                                 fft_size: int,
                                 buffer: vd.Buffer,
                                 kernel: vd.Buffer):
    vd.fft.fft(buffer)


def vkfft_test_function(config: AccuracyConfig,
                        fft_size: int,
                        buffer: vd.Buffer,
                        kernel: vd.Buffer):
    vd.vkfft.fft(buffer)


if __name__ == "__main__":
    run_accuracy_test("vkdispatch_accuracy", "vkdispatch", vkdispatch_fft_test_function)
    run_accuracy_test("vkfft_accuracy", "vkfft", vkfft_test_function)
