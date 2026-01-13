import glob
import csv
from typing import Dict, Tuple, Set, List
from matplotlib import pyplot as plt
import numpy as np
import sys
import os
import shutil
import matplotlib.colors as mcolors
import colorsys

# Nested structure:
# merged[backend][fft_size] = (mean, std)
MergedType = Dict[str, Dict[int, Tuple[float, float]]]

def adjust_lightness(color, factor):
    """Lighten or darken a given matplotlib color by multiplying its lightness by 'factor'."""
    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color
    r, g, b = mcolors.to_rgb(c)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, min(1, l * factor))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)

def get_backend_color(backend_name: str) -> Tuple[float, float, float]:
    color_base_nvidia = plt.cm.tab10(2)  # Green
    color_base_vulkan = plt.cm.tab10(3)  # Red
    color_base_torch = plt.cm.tab10(0)   # Blue
    color_base_zipfft = plt.cm.tab10(1)  # Orange

    if backend_name == "cufft":
        return adjust_lightness(color_base_nvidia, 0.8)
    elif backend_name == "cufftdx":
        return adjust_lightness(color_base_nvidia, 1.4)
    elif backend_name == "cufftdx_naive":
        return adjust_lightness(color_base_nvidia, 1.2)
    
    elif backend_name == "cufft_nvidia":
        return adjust_lightness(color_base_nvidia, 0.8)
    elif backend_name == "cufftdx_nvidia":
        return adjust_lightness(color_base_nvidia, 1.4)
    
    elif backend_name == "vkdispatch":
        return adjust_lightness(color_base_vulkan, 0.8)
    elif backend_name == "vkdispatch_transpose":
        return adjust_lightness(color_base_vulkan, 1.1)
    elif backend_name == "vkdispatch_naive":
        return adjust_lightness(color_base_vulkan, 0.6)
    elif backend_name == "vkfft":
        return adjust_lightness(color_base_vulkan, 1.4)
    elif backend_name == "vulkan":
        return adjust_lightness(color_base_vulkan, 1.0)
    
    elif backend_name == "torch":
        return adjust_lightness(color_base_torch, 0.8)
    
    elif backend_name == "zipfft":
        return adjust_lightness(color_base_zipfft, 1.6)
    elif backend_name == "zipfft_smem":
        return adjust_lightness(color_base_zipfft, 1.2)
    
    elif backend_name == "zipfft_transpose":
        return adjust_lightness(color_base_zipfft, 0.8)
    elif backend_name == "zipfft_transpose_smem":
        return adjust_lightness(color_base_zipfft, 0.6)
    elif backend_name == "zipfft_naive":
        return adjust_lightness(color_base_zipfft, 1.4)

    else:
        raise ValueError(f"Unknown backend name: {backend_name}")

def sort_backend(backends: Set[str]) -> List[str]:
    sorted_list = []
    
    if "vkdispatch" in backends:
        sorted_list.append("vkdispatch")
    if "vkdispatch_transpose" in backends:
        sorted_list.append("vkdispatch_transpose")
    if "vkdispatch_naive" in backends:
        sorted_list.append("vkdispatch_naive")
    if "vkfft" in backends:
        sorted_list.append("vkfft")
    if "vulkan" in backends:
        sorted_list.append("vulkan")

    if "cufft" in backends:
        sorted_list.append("cufft")
    if "cufftdx" in backends:
        sorted_list.append("cufftdx")
    if "cuda" in backends:
        sorted_list.append("cuda")

    if "cufftdx_naive" in backends:
        sorted_list.append("cufftdx_naive")

    if "cufft_nvidia" in backends:
        sorted_list.append("cufft_nvidia")
    if "cufftdx_nvidia" in backends:
        sorted_list.append("cufftdx_nvidia")

    if "torch" in backends:
        sorted_list.append("torch")
    if "zipfft" in backends:
        sorted_list.append("zipfft")

    if "zipfft_smem" in backends:
        sorted_list.append("zipfft_smem")

    if "zipfft_transpose" in backends:
        sorted_list.append("zipfft_transpose")

    if "zipfft_transpose_smem" in backends:
        sorted_list.append("zipfft_transpose_smem")
    if "zipfft_naive" in backends:
        sorted_list.append("zipfft_naive")

    return sorted_list

def read_bench_csvs(pattern) -> Tuple[MergedType, Set[str], Set[int]]:
    files = glob.glob(pattern)

    merged: MergedType = {}
    backends: Set[str] = set()
    fft_sizes: Set[int] = set()

    for filename in files:
        print(f'Reading: {filename}')

        if filename.endswith("merged.csv"):
            print('  Skipping merged.csv file')
            continue

        with open(filename, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                backend = row['Backend'].strip()
                size = int(row['FFT Size'])
                mean = float(row['Mean'])
                std = float(row['Std Dev'])

                backends.add(backend)
                fft_sizes.add(size)

                if backend not in merged:
                    merged[backend] = {}

                # last one wins if duplicates appear across files
                merged[backend][size] = (mean, std)

    return merged, backends, fft_sizes

def save_bar_graph(backends: List[str],
                           fft_sizes: List[int],
                           merged: MergedType,
                           outfile: str,
                           title: str,
                           xlabel: str,
                           ylabel: str,):
    # Choose the sizes to display
    used_fft_sizes = sorted(fft_sizes)

    x = np.arange(len(used_fft_sizes), dtype=float)
    n_backends = max(1, len(backends))
    width = 0.8 / n_backends  # total group width ~0.8

    plt.figure(figsize=(12, 6))

    for j, backend in enumerate(backends):
        # Center bars around tick: offsets in [-0.5..+0.5]*group_width
        xj = x + (j - (n_backends - 1) / 2) * width

        xs, heights, errs = [], [], []
        for i, size in enumerate(used_fft_sizes):
            entry = merged.get(backend, {}).get(size)
            if entry is None:
                # Skip if this backend didn't report this size
                continue
            mean, std = entry
            xs.append(xj[i])
            heights.append(mean)
            errs.append(std)

        if xs:
            plt.bar(xs, heights, width=width, yerr=errs, capsize=4, label=backend)

    plt.xticks(x, [str(s) for s in used_fft_sizes])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    print(f'Saved {outfile}')

def save_line_graph(
        backends: Set[str],
        fft_sizes: Set[int],
        merged: MergedType,
        outfile: str,
        title: str,
        xlabel: str,
        ylabel: str):
    plt.figure(figsize=(10, 6))

    used_fft_sizes = sorted(fft_sizes)

    for backend_name in sort_backend(backends):
        means = [
            merged[backend_name][i][0]
            for i in used_fft_sizes
        ]
        stds = [
            merged[backend_name][i][1]
            for i in used_fft_sizes
        ]
        
        plt.errorbar(
            used_fft_sizes,
            means,
            yerr=stds,
            label=backend_name,
            capsize=5,
            color=get_backend_color(backend_name)
        )
    plt.xscale('log', base=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(outfile)

def copy_files(src_dir: str, dst_dir: str, extensions: List[str] = None):
    """Copy files with specified extensions from src_dir to dst_dir."""
    
    if extensions is None:
        extensions = ['.csv', '.png']
    
    os.makedirs(dst_dir, exist_ok=True)
    
    for ext in extensions:
        pattern = os.path.join(src_dir, f'*{ext}')
        files = glob.glob(pattern)
        for src_file in files:
            dst_file = os.path.join(dst_dir, os.path.basename(src_file))
            shutil.copy2(src_file, dst_file)
            print(f'Copied: {src_file} -> {dst_file}')

def save_merged_csv(merged: MergedType, backends: Set[str], fft_sizes: Set[int], outfile: str):
    with open(outfile, "w", newline="") as f:
        fieldnames = ["Backend", "FFT Size", "Mean", "Std Dev"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for backend_name in backends:
            for size in fft_sizes:
                if size in merged[backend_name]:
                    mean, std = merged[backend_name][size]
                    writer.writerow({
                        "Backend": backend_name,
                        "FFT Size": size,
                        "Mean": mean,
                        "Std Dev": std,
                    })

def make_graph(test_name: str, title: str, xlabel: str, ylabel: str):
    src_dir = os.path.join("tests", test_name, "test_results")
    out_dir = os.path.join("test_results", test_name)

    copy_files(src_dir, out_dir)

    merged, backends, fft_sizes = read_bench_csvs(os.path.join(out_dir, "*.csv"))

    print('\nSummary:')
    print(f'Backends found: {sorted(backends)}')
    print(f'Convolution sizes found: {sorted(fft_sizes)}')
    print(f'Total entries: {sum(len(v) for v in merged.values())}')

    sorted_backends = sorted(backends)
    sorted_fft_sizes = sorted(fft_sizes)

    merged_filename = os.path.join(out_dir, "merged.csv")
    save_merged_csv(merged, sorted_backends, sorted_fft_sizes, merged_filename)

    graph_filename = os.path.join(out_dir, "graph.png")
    save_line_graph(
        sorted_backends,
        sorted_fft_sizes,
        merged,
        graph_filename,
        title,
        xlabel,
        ylabel
    )

    dst_graph = os.path.join("test_results", f"{test_name}_graph.png")

    shutil.copy2(graph_filename, dst_graph)
    print(f'Copied: {graph_filename} -> {dst_graph}')

if __name__ == '__main__':
    assert len(sys.argv) == 5, "Usage: python make_graph.py <data_directory> <title> <xlabel> <ylabel>"

    test_name = sys.argv[1]
    title = sys.argv[2]
    xlabel = sys.argv[3]
    ylabel = sys.argv[4]

    make_graph(test_name, title, xlabel, ylabel)
    
