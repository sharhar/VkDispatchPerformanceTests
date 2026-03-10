import figure_utils

tests = {
    "vkfft_naive": ("vkfft_vulkan", "conv_scaled_control"),
    "vkdispatch_vulkan": ("vkdispatch_vulkan", "conv_scaled_control"),
    "vkdispatch_naive_vulkan": ("vkdispatch_naive_vulkan", "conv_scaled_control"),
    "vkdispatch_cuda": ("vkdispatch_cuda", "conv_scaled_control"),
    "vkdispatch_naive_cuda": ("vkdispatch_naive_cuda", "conv_scaled_control"),
    "vkdispatch_opencl": ("vkdispatch_opencl", "conv_scaled_control"),
    "vkdispatch_naive_opencl": ("vkdispatch_naive_opencl", "conv_scaled_control"),
    "cufft": ("cufft", "conv_scaled_control"),
    "cufftdx": ("cufftdx", "conv_scaled_control"),
    "cufft_nvidia": ("cufft_nvidia", "conv_scaled_nvidia"),
    "cufftdx_nvidia": ("cufftdx_nvidia", "conv_scaled_nvidia"),
    "cufftdx_naive": ("cufftdx_naive", "conv_scaled_control")
}

test_data = figure_utils.load_tests(tests)

figure_utils.plot_data(
    test_data=test_data,
    scale_factor=3,
    output_name="fig1_scaled_nonstrided_convolution",
    #ncol=2,
    loc='lower right',
    split_y_axis=True
)