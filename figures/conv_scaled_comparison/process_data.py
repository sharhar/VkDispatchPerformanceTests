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

def get_test_ratios(dir_path: str, file1: str, file2: str) -> Dict[int, Tuple[float, float]]:
    data1 = get_test_data(os.path.join(dir_path, file1))
    data2 = get_test_data(os.path.join(dir_path, file2))

    assert data1.keys() == data2.keys(), "FFT sizes do not match between the two datasets."

    ratios_data = {}

    for size in data1.keys():
        datum1, std1 = data1[size]
        datum2, std2 = data2[size]

        ratio = datum1 / datum2
        ratio_std = ratio * ((std1 / datum1) ** 2 + (std2 / datum2) ** 2) ** 0.5

        ratios_data[size] = (ratio, ratio_std)
    
    return ratios_data

def write_test_data(filename: str, backend: str, data: Dict[int, Tuple[float, float]]):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Backend', 'FFT Size', 'Mean', 'Std Dev'])
        for size in sorted(data.keys()):
            ratio, std_dev = data[size]
            writer.writerow([backend, size, ratio, std_dev])


cufftdx_ratios = get_test_ratios(
    '../../conv_scaled_nvidia/test_results',
    'cufft.csv',
    'cufftdx.csv'
)
write_test_data('cufftdx.csv', 'cufftdx', cufftdx_ratios)

zipfft_ratios = get_test_ratios(
    '../../conv_scaled_control/test_results',
    'zipfft.csv',
    'cufft.csv'
)
write_test_data('zipfft.csv', 'zipfft', zipfft_ratios)

vkdispatch_ratios = get_test_ratios(
    '../../conv_scaled_control/test_results',
    'vkdispatch.csv',
    'cufft.csv'
)
write_test_data('vkdispatch.csv', 'vkdispatch', vkdispatch_ratios)