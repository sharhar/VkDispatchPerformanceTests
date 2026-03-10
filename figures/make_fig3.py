import figure_utils

tests = {
    "vkfft_naive": ("vkfft_naive_vulkan", "conv_2d"),
    "vkdispatch_naive_vulkan": ("vkdispatch_naive_vulkan", "conv_2d"),
    "vkdispatch_vulkan": ("vkdispatch_vulkan", "conv_2d_padded"),
    "vkdispatch_naive_cuda": ("vkdispatch_naive_cuda", "conv_2d"),
    "vkdispatch_cuda": ("vkdispatch_cuda", "conv_2d_padded"),
    "vkdispatch_naive_opencl": ("vkdispatch_naive_opencl", "conv_2d"),
    "vkdispatch_opencl": ("vkdispatch_opencl", "conv_2d_padded"),
    "cufft": ("cufft", "conv_2d_padded"),
    "cufftdx": ("cufftdx", "conv_2d_padded"),
    "cufftdx_naive": ("cufftdx_naive", "conv_2d_padded")
}

test_data = figure_utils.load_tests(tests)

figure_utils.plot_data(
    test_data=test_data,
    scale_factor=704/273,
    output_name="fig3_padded_2d_convolution",
    show_squared_x=True,
    max_fft_size=512,
    ncol=2
)