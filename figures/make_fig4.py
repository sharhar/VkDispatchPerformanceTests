import csv
from pathlib import Path
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt

import figure_utils

import figure_utils

tests = {
    "vkfft_accuracy": ("vkfft_accuracy_vulkan", "accuracy"),
    "vkdispatch_vulkan_accuracy": ("vkdispatch_accuracy_vulkan", "accuracy"),
    #"vkdispatch_cuda": ("vkdispatch_accuracy_cuda", "accuracy"),
    #"vkdispatch_opencl": ("vkdispatch_accuracy_opencl", "accuracy"),
    "cufft": ("cufft_accuracy", "accuracy"),
    "cufftdx_accuracy": ("cufftdx_accuracy", "accuracy"),
}

test_data = figure_utils.load_tests(tests)

figure_utils.plot_data(
    test_data=test_data,
    scale_factor=1,
    output_name="fig4_accuracy",
    max_fft_size=512,
    log_y=True,
    y_label="L2 Mean Error"
)
