import os
import csv
import dataclasses

import numpy as np
from matplotlib import pyplot as plt

from typing import Dict, Tuple, Set

# --- Data Loading ---

def get_test_data(filename: str) -> Dict[int, Tuple[float, float]]:
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

def load_test_by_name(test_name: str) -> Dict[int, Tuple[float, float]]:
    # Adjust path as necessary
    filename = f"../../test_results/conv_2d_padded/{test_name}.csv"
    return get_test_data(filename)

test_names = [
    #"vkfft",
    "vkdispatch",
    "vkdispatch_naive",
    #"cufft",
    #"cufftdx"
]

# --- Configuration ---

@dataclasses.dataclass
class TestProperties:
    name: str
    color: str
    marker: str
    y_scaling: float = 1.0  # Controls which Y-axis category the data belongs to

high_scale = 704.0/273.0

# Define properties with the specific 3x scaling for high-performance variants
test_properties = {
    "vkdispatch": TestProperties(
        name="VkDispatch (Fused)",
        color='blue',
        marker='o',
        y_scaling=high_scale
    ),
    "vkdispatch_naive": TestProperties(
        name="VkDispatch (Naive)",
        color='red',
        marker='s',
        y_scaling=1.0
    ),
    # "vkfft": TestProperties(
    #     name="VkFFT (Fused)",
    #     color='purple',
    #     marker='v',
    #     y_scaling=3.0
    # ),
    # "cufft": TestProperties(
    #     name="cuFFT",
    #     color='green',
    #     marker='^',
    #     y_scaling=1.0  # Standard
    # ),
    # "cufftdx": TestProperties(
    #     name="cuFFTDX",
    #     color='orange',
    #     marker='D',
    #     y_scaling=3.0  # High throughput
    # )
}

tests_data = {
    name: load_test_by_name(name)
    for name in test_names
}

# --- Plotting Setup ---

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

def extract_plot_data(data_dict):
    """Sorts data by FFT size and separates into x, y, and y_err arrays."""
    if not data_dict:
        return np.array([]), np.array([]), np.array([])
    sorted_keys = sorted(data_dict.keys())
    x = np.array(sorted_keys)
    y = np.array([data_dict[k][0] for k in sorted_keys])
    y_err = np.array([data_dict[k][1] for k in sorted_keys])
    return x, y, y_err

# --- Plotting Loop ---

all_sizes = set()

for test_name in test_names:
    if test_name not in tests_data or not tests_data[test_name]:
        continue

    props = test_properties[test_name]
    
    # 1. Load raw data
    x, y_raw, y_err_raw = extract_plot_data(tests_data[test_name])
    all_sizes.update(x)

    # 2. Normalize data to the primary axis by dividing by the scaling factor
    #    If scaling is 3.0, a value of 300 becomes 100, aligning it with unscaled data.
    y_plot = y_raw / props.y_scaling
    y_err_plot = y_err_raw / props.y_scaling

    ax.errorbar(x, y_plot,
                yerr=y_err_plot,
                label=props.name, 
                color=props.color,
                marker=props.marker,
                capsize=3, elinewidth=1, markersize=5,
                linestyle='-', linewidth=1.5, alpha=0.9)

# --- Axes formatting ---

# 1. Primary Axis (Left) - Standard Scale (1x)
ax.set_xscale('log', base=2)
ax.set_xlabel('FFT Size (N)')
ax.set_ylabel('Naive Bandwidth (GB/s)') 

# 2. Secondary Axis (Right) - High-Performance Scale (3x)
#    We clone the x-axis but create an independent y-axis
ax2 = ax.twinx()

#    IMPORTANT: Sync limits. We read the auto-scaled limits of the primary axis
#    and multiply them by 3.0 for the secondary axis. This ensures the grid lines
#    visually represent 1x on the left and 3x on the right.
y_min, y_max = ax.get_ylim()
ax2.set_ylim(y_min * high_scale, y_max * high_scale)
ax2.set_ylabel('Fused Bandwidth (GB/s)')

#    Turn off grid for the second axis to prevent misalignment clutter
#    (The primary grid now serves both axes correctly)
ax2.grid(False)

# X-ticks formatting
if all_sizes:
    sorted_sizes = sorted(list(all_sizes))
    ax.set_xticks(sorted_sizes)
    ax.set_xticklabels(sorted_sizes)

# Add grid and legend (Legend is attached to ax, so it contains all lines)
ax.grid(True, which="both", ls="-", alpha=0.3)
ax.legend(frameon=True, loc='best')

# --- Saving ---
output_filename = "2d_padded_convolution.pdf"
plt.tight_layout()
plt.savefig(output_filename, format='pdf', dpi=300)
print(f"Graph saved successfully to {output_filename}")