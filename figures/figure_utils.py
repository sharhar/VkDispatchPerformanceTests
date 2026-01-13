
import os
import csv
import dataclasses

import numpy as np
from matplotlib import pyplot as plt

from typing import Dict, Tuple, Set

def load_test_data(test_id: str, test_category: str) -> Dict[int, Tuple[float, float]]:
    filename = f"../test_results_ref2/{test_category}/{test_id}.csv"

    results = {}
    if not os.path.exists(filename):
        print(f"Warning: File not found: {filename}")
        return results

    with open(filename, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                size = int(row['FFT Size'])
                mean = float(row['Mean'])
                std = float(row['Std Dev'])
                results[size] = (mean, std)
            except (ValueError, KeyError) as e:
                print(f"Skipping malformed row in {filename}: {e}")
                continue
    
    return results

def load_tests(tests: Dict[str, Tuple[str, str]]):
    test_data = {}
    for test_name, (test_id, test_category) in tests.items():
        data = load_test_data(test_id, test_category)
        test_data[test_name] = data
    return test_data

@dataclasses.dataclass
class TestProperties:
    name: str
    color: str
    marker: str
    y_scaling: bool

# Define properties with the specific 3x scaling for high-performance variants
test_properties = {
    "vkfft": TestProperties(
        name="VkFFT (Fused)",
        color='purple',
        marker='v',
        y_scaling=True
    ),
    "vkfft_naive": TestProperties(
        name="VkFFT (Naive)",
        color='purple',
        marker='v',
        y_scaling=False
    ),
    "vkdispatch": TestProperties(
        name="VkDispatch (Fused)",
        color='blue',
        marker='o',
        y_scaling=True
    ),
    "vkdispatch_transpose": TestProperties(
        name="VkDispatch KT (Fused)",
        color='cyan',
        marker='o',
        y_scaling=True
    ),
    "vkdispatch_naive": TestProperties(
        name="VkDispatch (Naive)",
        color='red',
        marker='s',
        y_scaling=False
    ),
    "cufft": TestProperties(
        name="cuFFT",
        color='green',
        marker='^',
        y_scaling=None
    ),
    "cufft_nvidia": TestProperties(
        name="cuFFT NV (Naive)",
        color='green',
        marker='^',
        y_scaling=False
    ),
    "cufftdx": TestProperties(
        name="cuFFTDx (Fused)",
        color='orange',
        marker='D',
        y_scaling=True
    ),
    "cufftdx_nvidia": TestProperties(
        name="cuFFTDx NV (Fused)",
        color='orange',
        marker='D',
        y_scaling=True
    ),
    "cufftdx_naive": TestProperties(
        name="cuFFTDx (Naive)",
        color='orange',
        marker='D',
        y_scaling=False
    ),
}

def extract_plot_data(data_dict):
    """Sorts data by FFT size and separates into x, y, and y_err arrays."""
    if not data_dict:
        return np.array([]), np.array([]), np.array([])
    sorted_keys = sorted(data_dict.keys())
    x = np.array(sorted_keys)
    y = np.array([data_dict[k][0] for k in sorted_keys])
    y_err = np.array([data_dict[k][1] for k in sorted_keys])
    return x, y, y_err

def plot_data(test_data: Dict[str, Dict[int, Tuple[float, float]]],
              scale_factor: float,
              output_name: str,
              split_graphs: bool = False):
    plt.style.use('seaborn-v0_8-whitegrid')
        
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (12, 4) if split_graphs else (6, 4)
    })

    if split_graphs:
        fig, (ax_left, ax_right) = plt.subplots(1, 2)
        axes_pairs = [(ax_left, 128, "le"), (ax_right, 128, "gt")]
    else:
        fig, ax = plt.subplots()
        axes_pairs = [(ax, None, None)]

    for ax_main, threshold, mode in axes_pairs:
        all_sizes = set()

        for test_name in test_data.keys():
            props = test_properties[test_name]
            
            x, y_raw, y_err_raw = extract_plot_data(test_data[test_name])

            # Filter data based on threshold if splitting
            if threshold is not None:
                if mode == "le":
                    mask = x <= threshold
                else:  # mode == "gt"
                    mask = x > threshold
                x = x[mask]
                y_raw = y_raw[mask]
                y_err_raw = y_err_raw[mask]
            
            if len(x) == 0:
                continue

            all_sizes.update(x)

            # Determine y_scaling behavior
            # None means: scale on left graph, don't scale on right graph
            if props.y_scaling is None:
                do_scaling = (mode == "le") if split_graphs else False
            else:
                do_scaling = props.y_scaling

            y_plot = y_raw / scale_factor if do_scaling else y_raw
            y_err_plot = y_err_raw / scale_factor if do_scaling else y_err_raw

            ax_main.errorbar(x, y_plot,
                        yerr=y_err_plot,
                        label=props.name, 
                        color=props.color,
                        marker=props.marker,
                        capsize=3, elinewidth=1, markersize=5,
                        linestyle='-', linewidth=1.5, alpha=0.9)

        ax_main.set_xscale('log', base=2)
        ax_main.set_xlabel('FFT Size (N)')
        ax_main.set_ylabel('Naive Bandwidth (GB/s)') 

        ax2 = ax_main.twinx()
        y_min, y_max = ax_main.get_ylim()
        ax2.set_ylim(y_min * scale_factor, y_max * scale_factor)
        ax2.set_ylabel('Fused Bandwidth (GB/s)')
        ax2.grid(False)

        if all_sizes:
            sorted_sizes = sorted(list(all_sizes))
            ax_main.set_xticks(sorted_sizes)
            ax_main.set_xticklabels(sorted_sizes)

        ax_main.grid(True, which="both", ls="-", alpha=0.3)
        ax_main.legend(frameon=True, loc='best')

    plt.tight_layout()
    plt.savefig(f"{output_name}.pdf", format='pdf', dpi=300)
    print(f"Graph saved successfully to {output_name}.pdf")

    plt.savefig(f"{output_name}.png", format='png', dpi=300)
    print(f"Graph saved successfully to {output_name}.png")

def plot_data_legacy(test_data: Dict[str, Dict[int, Tuple[float, float]]],
              scale_factor: float,
              output_name: str):
    plt.style.use('seaborn-v0_8-whitegrid')
        
    # Update font sizes for readability in LaTeX
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (6, 4) # Standard single-column width
    })

    fig, ax = plt.subplots()
    all_sizes = set()

    for test_name in test_data.keys():
        props = test_properties[test_name]
        
        # 1. Load raw data
        x, y_raw, y_err_raw = extract_plot_data(test_data[test_name])
        all_sizes.update(x)

        # 2. Normalize data to the primary axis by dividing by the scaling factor
        #    If scaling is 3.0, a value of 300 becomes 100, aligning it with unscaled data.
        
        y_plot = y_raw / scale_factor if props.y_scaling else y_raw
        y_err_plot = y_err_raw / scale_factor if props.y_scaling else y_err_raw

        ax.errorbar(x, y_plot,
                    yerr=y_err_plot,
                    label=props.name, 
                    color=props.color,
                    marker=props.marker,
                    capsize=3, elinewidth=1, markersize=5,
                    linestyle='-', linewidth=1.5, alpha=0.9)

    ax.set_xscale('log', base=2)
    ax.set_xlabel('FFT Size (N)')
    ax.set_ylabel('Naive Bandwidth (GB/s)') 

    ax2 = ax.twinx()

    y_min, y_max = ax.get_ylim()
    ax2.set_ylim(y_min * scale_factor, y_max * scale_factor)
    ax2.set_ylabel('Fused Bandwidth (GB/s)')

    ax2.grid(False)

    if all_sizes:
        sorted_sizes = sorted(list(all_sizes))
        ax.set_xticks(sorted_sizes)
        ax.set_xticklabels(sorted_sizes)

    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend(frameon=True, loc='best')

    plt.tight_layout()
    plt.savefig(f"{output_name}.pdf", format='pdf', dpi=300)
    print(f"Graph saved successfully to {output_name}.pdf")

    plt.savefig(f"{output_name}.png", format='png', dpi=300)
    print(f"Graph saved successfully to {output_name}.png")