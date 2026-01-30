import figure_utils

tests = {
    "vkfft_naive": ("vkfft", "conv_scaled_control"),
    "vkdispatch": ("vkdispatch", "conv_scaled_control"),
    "vkdispatch_naive": ("vkdispatch_naive", "conv_scaled_control"),
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