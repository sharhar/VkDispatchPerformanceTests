import figure_utils

tests = {
    "vkfft": ("vkfft", "conv_2d"),
    "vkdispatch": ("vkdispatch", "conv_2d"),
    "vkdispatch_naive": ("vkdispatch_naive", "conv_2d"),
    "vkdispatch_transpose": ("vkdispatch_transpose", "conv_2d"),
    "cufft": ("cufft", "conv_2d"),
    "cufftdx": ("cufftdx", "conv_2d")
}

test_data = figure_utils.load_tests(tests)

figure_utils.plot_data(
    test_data=test_data,
    scale_factor=11/7,
    output_name="fig2_2d_convolution",
    split_graphs=True
)