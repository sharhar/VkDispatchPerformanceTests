import figure_utils

tests = {
    "vkfft": ("vkfft", "conv_2d_padded"),
    "vkdispatch": ("vkdispatch", "conv_2d_padded"),
    "vkdispatch_naive": ("vkdispatch_naive", "conv_2d_padded"),
    "vkdispatch_transpose": ("vkdispatch_transpose", "conv_2d_padded"),
    #"cufft": ("cufft", "conv_2d_padded"),
    "cufftdx": ("cufftdx", "conv_2d_padded")
}

test_data = figure_utils.load_tests(tests)

figure_utils.plot_data(
    test_data=test_data,
    scale_factor=704/273,
    output_name="fig3_padded_2d_convolution",
    split_graphs=True
)