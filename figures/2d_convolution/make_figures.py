import os
import csv

from typing import Dict, Tuple, Set

def get_test_data(filename: str) -> Dict[int, Tuple[float, float]]:
    results = {}

    with open(filename, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            size = int(row['FFT Size'])
            mean = float(row['Mean'])
            std = float(row['Std Dev'])
            results[size] = (mean, std)
    
    return results

vkdispatch_data = get_test_data('../../test_results/conv_2/vkdispatch.csv')
vkdispatch_naive_data = get_test_data('../../test_results/conv_2/vkdispatch_naive.csv')
vkdispatch_transpose_data = get_test_data('../../test_results/conv_2/vkdispatch_transpose.csv')
zipfft_data = get_test_data('../../test_results/conv_2/zipfft.csv')
zipfft_transpose_data = get_test_data('../../test_results/conv_2/zipfft_transpose.csv')
cufft_data = get_test_data('../../test_results/conv_2/cufft.csv')
