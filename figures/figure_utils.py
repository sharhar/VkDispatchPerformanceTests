
import os
import csv
import dataclasses

import numpy as np
from matplotlib import pyplot as plt

import sys

from typing import Dict, List, Optional, Tuple, Union

SingleTestDataType = Dict[int, Tuple[float, float]]
DualTestDataType = Tuple[SingleTestDataType, SingleTestDataType]

TestDataType = Union[
   SingleTestDataType,
   DualTestDataType
]

def load_test_data(test_id: str, test_category: str, load_dual: bool, test_folder: str = None) -> TestDataType:
    if load_dual:
        assert test_folder is None, "Dual test loading requires explicit test_folder argument"

        return (
            load_test_data(test_id, test_category, False, test_folder="../test_results_nvidia"),
            load_test_data(test_id, test_category, False, test_folder="../test_results_macos")
        )

    if test_folder is None:
        test_folder = "../test_results"

    filename = f"{test_folder}/{test_category}/{test_id}.csv"

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
    load_dual = sys.argv.count("--dual_nvidia_macos") != 0
    test_data = {}
    for test_name, (test_id, test_category) in tests.items():
        data = load_test_data(test_id, test_category, load_dual)
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
#   Blue:      #0072B2 - not used
#   Orange:    #E69F00 - used for VkDispatch KT
#   Sky Blue:  #56B4E9 - used for VkFFT
#   Vermillion:#D55E00 - used for VkDispatch
#   Teal:      #009E73 - used for cuFFTDx
#   Yellow:    #F0E442 - not used
#   Purple:    #CC79A7 - pinkish purple, used for cuFFTDx NV 
#   Black:     #000000 - CUDA
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
    "vkfft_accuracy": TestProperties(
        name="VkFFT",
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
    "vkdispatch_vulkan": TestProperties(
        name="VkDispatch VK (Fused)",
        color='#D55E00',
        marker='s',
        linestyle='-',
        y_scaling=True
    ),
    "vkdispatch_vulkan_accuracy": TestProperties(
        name="VkDispatch",
        color='#D55E00',
        marker='s',
        linestyle='-',
        y_scaling=True
    ),
    "vkdispatch_naive_vulkan": TestProperties(
        name="VkDispatch VK (Naive)",
        color='#D55E00',
        marker='o',
        linestyle='--',
        y_scaling=False
    ),

    "vkdispatch_cuda": TestProperties(
        name="VkDispatch CU (Fused)",
        color='#E69F00',
        marker='s', 
        linestyle='-',
        y_scaling=True
    ),
    "vkdispatch_naive_cuda": TestProperties(
        name="VkDispatch CU (Naive)",
        color='#E69F00',
        marker='o', 
        linestyle='--',
        y_scaling=False
    ),

    "vkdispatch_opencl": TestProperties(
        name="VkDispatch CL (Fused)",
        color='#0072B2',
        marker='s', 
        linestyle='-',
        y_scaling=True
    ),
    "vkdispatch_naive_opencl": TestProperties(
        name="VkDispatch CL (Naive)",
        color='#0072B2',
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
        color='#009E73',
        marker='v',
        linestyle='-',
        y_scaling=True
    ),
    "cufftdx_accuracy": TestProperties(
        name="cuFFTDx",
        color='#009E73',
        marker='v',
        linestyle='-',
        y_scaling=True
    ),
    "cufftdx_naive": TestProperties(
        name="cuFFTDx (Naive)",
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
    
    return (category, label)


def sort_legend(ax):
    """Sorts legend: Naive on top, cuFFT reference middle, Fused below."""
    handles, labels = ax.get_legend_handles_labels()

    # Zip, sort, unzip
    sorted_pairs = sorted(zip(handles, labels), key=lambda x: get_legend_sort_key(x[1]))
    sorted_handles, sorted_labels = zip(*sorted_pairs) if sorted_pairs else ([], [])

    return list(sorted_handles), list(sorted_labels)

def sort_legend_from_axes(axes):
    handles_by_label = {}

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            handles_by_label[label] = handle

    sorted_pairs = sorted(handles_by_label.items(), key=lambda x: get_legend_sort_key(x[0]))
    if not sorted_pairs:
        return [], []

    sorted_labels = [label for label, _ in sorted_pairs]
    sorted_handles = [handle for _, handle in sorted_pairs]
    return sorted_handles, sorted_labels

def normalize_test_data_panels(test_data: Dict[str, TestDataType]) -> Tuple[List[Tuple[str, Optional[str], Dict[str, SingleTestDataType]]], bool]:
    dual_mode = any(isinstance(data, tuple) for data in test_data.values())

    if not dual_mode:
        return [("single", None, test_data)], False

    nvidia_data = {}
    macos_data = {}

    for test_name, data in test_data.items():
        if isinstance(data, tuple):
            nvidia_data[test_name], macos_data[test_name] = data
        else:
            nvidia_data[test_name] = data
            macos_data[test_name] = data

    return [
        ("nvidia", "NVIDIA", nvidia_data),
        ("macos", "macOS", macos_data),
    ], True

def configure_axis_ticks(ax_main, all_sizes, show_squared_x: bool):
    if not all_sizes:
        return

    sorted_sizes = sorted(list(all_sizes))
    ax_main.set_xticks(sorted_sizes)

    if show_squared_x:
        labels = [f"${s}^2$" for s in sorted_sizes]
        ax_main.set_xticklabels(labels)
    else:
        ax_main.set_xticklabels(sorted_sizes)

def plot_panel(ax_main,
               panel_test_data: Dict[str, SingleTestDataType],
               scale_factor: float,
               split_y_axis: bool,
               show_squared_x: bool,
               log_y: bool,
               y_label: Optional[str]):
    all_sizes = set()

    for test_name in panel_test_data.keys():
        props = test_properties[test_name]

        x, y_raw, y_err_raw = extract_plot_data(panel_test_data[test_name])

        if len(x) == 0:
            continue

        all_sizes.update(x)

        if props.y_scaling is None:
            do_scaling = False
        else:
            do_scaling = props.y_scaling and split_y_axis

        y_plot = y_raw / scale_factor if do_scaling else y_raw
        y_err_plot = y_err_raw / scale_factor if do_scaling else y_err_raw

        ax_main.plot(x, y_plot,
                    label=props.name,
                    color=props.color,
                    marker=props.marker,
                    markersize=5,
                    linestyle=props.linestyle, linewidth=1, alpha=0.9)

    ax_main.set_xscale('log', base=2)
    ax_main.set_xlabel('FFT Size (N)')

    if split_y_axis:
        ax_main.set_ylabel('Naive Effective Bandwidth (GB/s)')

        ax2 = ax_main.twinx()
        y_min, y_max = ax_main.get_ylim()
        ax2.set_ylim(y_min * scale_factor, y_max * scale_factor)
        ax2.set_ylabel('Fused Effective Bandwidth (GB/s)')
        ax2.grid(False)
    else:
        if y_label is not None:
            ax_main.set_ylabel(y_label)
        else:
            ax_main.set_ylabel('Effective Bandwidth (GB/s)')

    configure_axis_ticks(ax_main, all_sizes, show_squared_x)

    if log_y:
        ax_main.set_yscale("log")

    ax_main.grid(True, which="both", ls="-", alpha=0.3)

def save_single_plot_data_csv(test_data: Dict[str, SingleTestDataType], output_name: str):
    """
    Saves the aggregated test data to a CSV file.
    Rows are aligned by FFT Size. Columns use the human-readable names.
    """
    # 1. Collect all unique FFT sizes across all tests
    all_sizes = set()
    for data in test_data.values():
        all_sizes.update(data.keys())
    sorted_sizes = sorted(list(all_sizes))

    # 2. Prepare headers using human-readable names from test_properties
    # We sort keys to ensure column order is deterministic
    sorted_test_keys = sorted(test_data.keys())
    
    headers = ['FFT Size']
    for key in sorted_test_keys:
        nice_name = test_properties[key].name
        headers.extend([f"{nice_name} Mean", f"{nice_name} Std"])

    # 3. Write to CSV
    csv_filename = f"{output_name}.csv"
    try:
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for size in sorted_sizes:
                row = [size]
                for key in sorted_test_keys:
                    # check if this specific test has data for this FFT size
                    if size in test_data[key]:
                        mean, std = test_data[key][size]
                        row.extend([mean, std])
                    else:
                        # Empty strings for missing data points
                        row.extend(["", ""]) 
                writer.writerow(row)
        print(f"Data saved successfully to {csv_filename}")
    except IOError as e:
        print(f"Error saving CSV {csv_filename}: {e}")

def save_plot_data_csv(test_data: Dict[str, TestDataType], output_name: str):
    panel_data, dual_mode = normalize_test_data_panels(test_data)

    if not dual_mode:
        save_single_plot_data_csv(panel_data[0][2], output_name)
        return

    for panel_key, _, panel_test_data in panel_data:
        save_single_plot_data_csv(panel_test_data, f"{output_name}_{panel_key}")

def plot_data(test_data: Dict[str, TestDataType],
              scale_factor: float,
              output_name: str,
              split_y_axis: bool = False,
              ncol: int = 1,
              loc: str = 'best',
              fontsize: int = 8,
              show_squared_x: bool = False,
              max_fft_size: int = None,
              log_y: bool = False,
              y_label: str = None):
    plt.style.use('seaborn-v0_8-whitegrid')
        
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 12,
        'figure.figsize': (6, 4)
    })

    panel_data, dual_mode = normalize_test_data_panels(test_data)

    if dual_mode:
        fig, all_axes = plt.subplots(
            1, 3,
            figsize=(15, 4.5),
            gridspec_kw={'width_ratios': [1, 1, 0.8]}
        )
        axes = np.atleast_1d(all_axes[:2])
        legend_ax = all_axes[2]
    else:
        fig, all_axes = plt.subplots(
            1, 2,
            figsize=(10, 4.5),
            gridspec_kw={'width_ratios': [1.6, 0.9]}
        )
        axes = [all_axes[0]]
        legend_ax = all_axes[1]

    legend_ax.axis('off')

    # final_average, fused_average = save_data_average(test_data, max_fft_size=max_fft_size, scale_factor=scale_factor, output_name=output_name)

    for ax_main, (_, panel_title, panel_test_data) in zip(axes, panel_data):
        plot_panel(
            ax_main=ax_main,
            panel_test_data=panel_test_data,
            scale_factor=scale_factor,
            split_y_axis=split_y_axis,
            show_squared_x=show_squared_x,
            log_y=log_y,
            y_label=y_label
        )

        if panel_title is not None:
            ax_main.set_title(panel_title)

    handles, labels = sort_legend_from_axes(axes)
    if handles:
        legend_ax.legend(
            handles,
            labels,
            frameon=True,
            loc='center',
            ncol=1,
            fontsize=max(fontsize + 2, 12),
            borderpad=1.2,
            labelspacing=1.1,
            handlelength=2.4,
            handletextpad=0.8
        )

    plt.tight_layout()

    plt.savefig(f"{output_name}.pdf", format='pdf', dpi=300)
    print(f"Graph saved successfully to {output_name}.pdf")

    plt.savefig(f"{output_name}.png", format='png', dpi=300)
    print(f"Graph saved successfully to {output_name}.png")

    save_plot_data_csv(test_data, output_name)
