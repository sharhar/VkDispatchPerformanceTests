import glob
import csv
from typing import Dict, Tuple, Set, List
from matplotlib import pyplot as plt
import numpy as np
import sys
import os
import shutil

# Nested structure:
# merged[backend][fft_size] = (mean, std)
MergedType = Dict[str, Dict[int, Tuple[float, float]]]

def read_bench_csvs(pattern) -> Tuple[MergedType, Set[str], Set[int]]:
    files = glob.glob(pattern)

    merged: MergedType = {}
    backends: Set[str] = set()
    fft_sizes: Set[int] = set()

    for filename in files:
        print(f'Reading: {filename}')
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

def save_grouped_bar_graph(backends: List[str],
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

    # X axis as categorical sizes (more readable for grouped bars)
    plt.xticks(x, [str(s) for s in used_fft_sizes])
    #plt.xlabel('Convolution Size (FFT size)')
    #plt.ylabel('GB/s (higher is better)')
    #plt.title('Scaled Nonstrided Convolution Performance Comparison')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    print(f'Saved {outfile}')

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
    src_dir = os.path.join(test_name, "test_results")
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
    save_grouped_bar_graph(
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
    
