
import os
import csv
import dataclasses

import numpy as np
from matplotlib import pyplot as plt

from typing import Dict, Tuple, Set

def load_test_data(test_id: str, test_category: str) -> Dict[int, Tuple[float, float]]:
    filename = f"../test_results/{test_category}/{test_id}.csv"

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
    linestyle: str
    y_scaling: bool

# Colorblind-safe palette (Okabe-Ito):
#   Blue:      #0072B2
#   Orange:    #E69F00
#   Sky Blue:  #56B4E9
#   Vermillion:#D55E00
#   Teal:      #009E73
#   Yellow:    #F0E442
#   Purple:    #CC79A7
#   Black:     #000000
#
# Line styles:
#   Naive  → '--' (dashed)
#   Fused  → '-'  (solid)
#   Ref    → ':'  (dotted)
#
# 'o' (circle)
# 's' (square)
# '^' (triangle up)
# 'v' (triangle down)
# 'D' (diamond)
# 'p' (pentagon)
# '*' (star)
# 'X' (x)
# 'P' (plus filled)
# 'h' (hexagon1)

test_properties = {
    # === VkFFT family ===
    "vkfft": TestProperties(
        name="VkFFT (Fused)",
        color="#56B4E9",
        marker='P',
        linestyle='-',
        y_scaling=True
    ),
    "vkfft_naive": TestProperties(
        name="VkFFT (Naive)",
        color='#56B4E9',
        marker='h',
        linestyle='--',
        y_scaling=False
    ),

    # === VkDispatch family ===
    "vkdispatch": TestProperties(
        name="VkDispatch (Fused)",
        color='#D55E00',
        marker='s',
        linestyle='-',
        y_scaling=True
    ),
    "vkdispatch_transpose": TestProperties(
        name="VkDispatch KT (Fused)",
        color='#E69F00',
        marker='D', 
        linestyle='-',
        y_scaling=True
    ),
    "vkdispatch_naive": TestProperties(
        name="VkDispatch (Naive)",
        color='#D55E00',
        marker='o',
        linestyle='--',
        y_scaling=False
    ),

    # === cuFFT family ===
    "cufft": TestProperties(
        name="cuFFT",
        color='#000000',      # black (reference baseline)
        marker='p',
        linestyle=':',
        y_scaling=None
    ),
    "cufft_nvidia": TestProperties(
        name="cuFFT NV (Naive)",
        color='#CC79A7',
        marker='*',
        linestyle='--',
        y_scaling=False
    ),
    "cufftdx_nvidia": TestProperties(
        name="cuFFTDx NV (Fused)",
        color='#CC79A7',
        marker='X',
        linestyle='-',
        y_scaling=True
    ),

    # === cuFFTDx family ===
    "cufftdx": TestProperties(
        name="cuFFTDx (Fused)",
        #color='#CC79A7',
        color='#009E73',
        marker='v',
        linestyle='-',
        y_scaling=True
    ),
    "cufftdx_naive": TestProperties(
        name="cuFFTDx (Naive)",
        #color='#CC79A7',
        color='#009E73',
        marker='^',
        linestyle='--',
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

def get_legend_sort_key(label: str) -> tuple:
    """
    Returns a sort key tuple: (category, name)
    Category: 0 = Naive, 1 = Reference (cuFFT), 2 = Fused
    """
    label_lower = label.lower()

    if "vkdispatch" in label_lower:
        category = 0
    elif "cufftdx" in label_lower:
        category = 1
    elif "cufft" in label_lower and "nv" in label_lower:
        category = 2
    elif "vkfft" in label_lower:
        category = 3
    elif "cufft" == label_lower:
        category = 4

    print(f"Assigned sort category {category} to label: {label}")
    
    return (category, label)


def sort_legend(ax):
    """Sorts legend: Naive on top, cuFFT reference middle, Fused below."""
    handles, labels = ax.get_legend_handles_labels()
    
    # Zip, sort, unzip
    sorted_pairs = sorted(zip(handles, labels), key=lambda x: get_legend_sort_key(x[1]))
    sorted_handles, sorted_labels = zip(*sorted_pairs) if sorted_pairs else ([], [])
    
    return list(sorted_handles), list(sorted_labels)

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

            # Plot without error bars for clarity
            ax_main.plot(x, y_plot,
                        label=props.name, 
                        color=props.color,
                        marker=props.marker,
                        markersize=5,
                        linestyle=props.linestyle, linewidth=1, alpha=0.9)

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
        #ax_main.legend(frameon=True, loc='best')
        handles, labels = sort_legend(ax_main)
        ax_main.legend(handles, labels, frameon=True, loc='best')

    plt.tight_layout()
    plt.savefig(f"{output_name}.pdf", format='pdf', dpi=300)
    print(f"Graph saved successfully to {output_name}.pdf")

    plt.savefig(f"{output_name}.png", format='png', dpi=300)
    print(f"Graph saved successfully to {output_name}.png")