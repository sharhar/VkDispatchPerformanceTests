import csv
from pathlib import Path
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt

import figure_utils


def load_accuracy_data(filename: Path) -> Dict[str, np.ndarray]:
    results = {
        "size": np.array([], dtype=int),
        "mean": np.array([], dtype=float),
        "std": np.array([], dtype=float),
        "worst_rel": np.array([], dtype=float),
    }

    if not filename.exists():
        print(f"Warning: File not found: {filename}")
        return results

    sizes = []
    means = []
    stds = []
    worst_relative = []

    with filename.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                sizes.append(int(row["FFT Size"]))
                means.append(float(row["Mean"]))
                stds.append(float(row["Std Dev"]))
                worst_relative.append(float(row["Worst Max Relative Error"]))
            except (ValueError, KeyError) as error:
                print(f"Skipping malformed row in {filename}: {error}")
                continue

    if not sizes:
        return results

    order = np.argsort(np.array(sizes))
    results["size"] = np.array(sizes, dtype=int)[order]
    results["mean"] = np.array(means, dtype=float)[order]
    results["std"] = np.array(stds, dtype=float)[order]
    results["worst_rel"] = np.array(worst_relative, dtype=float)[order]
    return results


def make_fig4():
    root_dir = Path(__file__).resolve().parents[1]
    data_files = {
        "vkdispatch": root_dir / "test_results" / "fft_nonstrided" / "vkdispatch_accuracy.csv",
        "vkfft": root_dir / "test_results" / "fft_nonstrided" / "vkfft_accuracy.csv",
        "cufft": root_dir / "test_results" / "fft_nonstrided" / "cufft_accuracy.csv",
        "cufftdx": root_dir / "test_results" / "fft_nonstrided" / "cufftdx_accuracy.csv",
    }

    test_data = {
        backend: load_accuracy_data(filename)
        for backend, filename in data_files.items()
    }

    if not any(data["size"].size for data in test_data.values()):
        raise RuntimeError("No accuracy data found in test_results/fft_nonstrided/*_accuracy.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (6.5, 6),
    })

    fig, (ax_mean, ax_worst) = plt.subplots(2, 1, sharex=True)

    all_sizes = set()
    for test_name in ("vkdispatch", "vkfft", "cufft", "cufftdx"):
        data = test_data.get(test_name)
        if data is None or data["size"].size == 0:
            continue

        props = figure_utils.test_properties[test_name]
        x = data["size"]
        mean = data["mean"]
        std = data["std"]
        worst_rel = data["worst_rel"]

        lower = np.maximum(mean - std, np.finfo(float).tiny)
        upper = np.maximum(mean + std, np.finfo(float).tiny)

        ax_mean.plot(
            x,
            mean,
            label=props.name,
            color=props.color,
            marker=props.marker,
            markersize=5,
            linestyle=props.linestyle,
            linewidth=1,
            alpha=0.9,
        )
        ax_mean.fill_between(x, lower, upper, color=props.color, alpha=0.15, linewidth=0)

        ax_worst.plot(
            x,
            worst_rel,
            label=props.name,
            color=props.color,
            marker=props.marker,
            markersize=5,
            linestyle=props.linestyle,
            linewidth=1,
            alpha=0.9,
        )

        all_sizes.update(x.tolist())

    ax_mean.set_yscale("log")
    ax_mean.set_ylabel("Relative L2 Error")
    ax_mean.set_title("FFT Accuracy (Nonstrided)")
    ax_mean.grid(True, which="both", ls="-", alpha=0.3)
    handles, labels = figure_utils.sort_legend(ax_mean)
    ax_mean.legend(handles, labels, frameon=True, loc="upper left")

    ax_worst.set_xscale("log", base=2)
    ax_worst.set_yscale("log")
    ax_worst.set_xlabel("FFT Size (N)")
    ax_worst.set_ylabel("Worst Max Relative Error")
    ax_worst.grid(True, which="both", ls="-", alpha=0.3)

    if all_sizes:
        sorted_sizes = sorted(all_sizes)
        ax_worst.set_xticks(sorted_sizes)
        ax_worst.set_xticklabels(sorted_sizes)

    plt.tight_layout()

    output_name = "fig4_fft_nonstrided_accuracy"
    plt.savefig(f"{output_name}.pdf", format="pdf", dpi=300)
    print(f"Graph saved successfully to {output_name}.pdf")

    plt.savefig(f"{output_name}.png", format="png", dpi=300)
    print(f"Graph saved successfully to {output_name}.png")


if __name__ == "__main__":
    make_fig4()
