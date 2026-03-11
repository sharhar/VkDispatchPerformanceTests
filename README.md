# VkDispatchPerformanceTests

This repository contains benchmark and accuracy tests for `vkdispatch`, plus scripts for turning benchmark outputs into publication-style figures. The primary user-facing entrypoint for running benchmarks is `run_all_tests.py`. Figure generation is handled separately from the checked-in data under `figures/`.

The repository already includes generated outputs under `test_results/` and `figures/`, so you can inspect the current data and remake the figures without rerunning the benchmarks.

## Repository layout

- `run_all_tests.py`: main benchmark and accuracy runner
- `tests/`: per-suite source code and raw per-run output directories
- `test_results/`: consolidated benchmark and accuracy CSVs plus quick-look graphs
- `figures/`: scripts for final figure generation and the rendered figure assets
- `dependencies/`: downloaded third-party CUDA dependencies created by the test scripts on first run

`make_graph.py` exists in the repository, but it is an internal helper called from `run_all_tests.py`. It is not intended to be used as a standalone user entrypoint.

## Prerequisites

Required tools:

- Python 3
- `git`
- `wget`
- `tar`
- `bash`

Required Python packages:

- `vkdispatch`
- `numpy`
- `matplotlib`

Backend-specific requirements:

- Vulkan benchmarks require a working Vulkan runtime and driver stack
- OpenCL benchmarks require a working OpenCL runtime and driver stack
- CUDA benchmarks require a CUDA toolkit installation with `nvcc`
- `run_all_tests.py` checks `CUDA_HOME/bin/nvcc` first if `CUDA_HOME` is set, otherwise it uses `nvcc` from `PATH`

Platform notes:

- cuFFT/cuFFTDx compilation is skipped on macOS by the current scripts
- CUDA accuracy helpers and cuFFTDx benchmarks rely on downloaded NVIDIA dependencies in `dependencies/`

## Running the tests

`run_all_tests.py` is the main entrypoint for benchmark and accuracy generation.

Examples:

```bash
python3 run_all_tests.py --vulkan --opencl --cuda
python3 run_all_tests.py --vulkan
python3 run_all_tests.py --opencl
python3 run_all_tests.py --cuda
python3 run_all_tests.py --validate --cuda
```

What the flags do:

- `--vulkan`: runs the Vulkan-backed `vkdispatch` path and `vkfft` where a `vkfft_test.py` exists
- `--opencl`: runs the OpenCL-backed `vkdispatch` path
- `--cuda`: runs the CUDA-backed `vkdispatch` path and the cuFFT/cuFFTDx comparisons
- `--validate`: runs validation-only cuFFTDx/cuFFT paths instead of the normal throughput benchmarks

Current built-in benchmark settings in `run_all_tests.py`:

```python
DATA_SIZE = 2**27
ITER_COUNT = 200
BATCH_SIZE = 20
REPEATS = 3
```

Suites currently enabled in the `__main__` block:

- `accuracy`
- `conv_scaled_nvidia` when `--cuda` is enabled
- `conv_scaled_control`
- `conv_2d`
- `conv_2d_padded`

Notes about first-time setup:

- On the first run, `run_all_tests.py` creates `dependencies/` and downloads:
  - NVIDIA MathDx (`nvidia-mathdx-25.06.1-cuda12.tar.gz`)
  - `cutlass` pinned to commit `e6e2cc29f5e7611dfc6af0ed6409209df0068cf2`
- The NVIDIA convolution helper under `tests/conv_scaled_nvidia/` also expects `dependencies/CUDALibrarySamples` on a fresh setup
- Raw benchmark outputs are first written into `tests/<suite>/test_results/`, then copied and merged into the top-level `test_results/` tree

## Making the figures

The final figures are generated from the consolidated root-level data under `test_results/`.

Run:

```bash
cd figures
bash make_figures.sh
```

This working directory matters. The figure scripts load data using paths like `../test_results/...`, so they must be run from inside `figures/`.

Figure outputs:

- `figures/*.pdf`
- `figures/*.png`
- `figures/*.csv`

The checked-in `test_results/` directory is enough to regenerate the figures. You do not need to rerun the benchmarks if you only want to remake `fig1` through `fig4`.

## Data layout

There are three main layers of generated data in this repository.

### 1. Raw suite outputs

Each suite writes its immediate outputs under:

- `tests/<suite>/test_results/`

These are the first CSVs emitted by the benchmark or accuracy scripts. They are suite-local working outputs.

### 2. Consolidated benchmark and accuracy outputs

`run_all_tests.py` copies suite-local results into:

- `test_results/<suite>/`

This directory contains the consolidated CSVs used by the figure scripts, including:

- backend-specific CSVs such as `test_results/conv_scaled_control/vkdispatch_vulkan.csv`
- suite-level `merged.csv`
- suite-level `graph.png`

It also writes top-level quick-look graphs such as:

- `test_results/conv_scaled_control_graph.png`
- `test_results/accuracy_graph.png`

### 3. Final figure assets

The scripts under `figures/` read from `../test_results/...` and write final publication-style assets into:

- `figures/*.pdf`
- `figures/*.png`
- `figures/*.csv`

## CSV formats

### Benchmark backend CSVs

Examples:

- `test_results/conv_scaled_control/vkdispatch_vulkan.csv`
- `test_results/conv_scaled_nvidia/cufftdx_nvidia.csv`

Format:

```text
Backend,FFT Size,Run 1 (GB/s),Run 2 (GB/s),...,Mean,Std Dev
```

Meaning:

- `Backend`: backend or implementation identifier
- `FFT Size`: transform or convolution size tested
- `Run N (GB/s)`: per-repeat throughput
- `Mean`: mean throughput over repeats
- `Std Dev`: standard deviation over repeats

### Suite merged CSVs

Examples:

- `test_results/conv_scaled_control/merged.csv`
- `test_results/accuracy/merged.csv`

Format:

```text
Backend,FFT Size,Mean,Std Dev
```

These files collapse the per-backend CSVs into one table per suite for plotting and inspection.

### Accuracy CSVs

Examples:

- `test_results/accuracy/vkdispatch_accuracy_vulkan.csv`
- `test_results/accuracy/cufft_accuracy.csv`

Format:

```text
Backend,FFT Size,Run 1 (Relative L2 Error),Run 2 (Relative L2 Error),...,Mean,Std Dev,Worst Max Relative Error,Worst Max Absolute Error
```

Meaning:

- `Run N (Relative L2 Error)`: per-repeat relative L2 error against the NumPy reference
- `Mean`: mean relative L2 error across repeats
- `Std Dev`: standard deviation of the relative L2 error
- `Worst Max Relative Error`: worst pointwise relative error observed across repeats
- `Worst Max Absolute Error`: worst pointwise absolute error observed across repeats

### Figure export CSVs

Examples:

- `figures/fig1_scaled_nonstrided_convolution.csv`
- `figures/fig4_accuracy.csv`

Format:

```text
FFT Size,<Series 1> Mean,<Series 1> Std,<Series 2> Mean,<Series 2> Std,...
```

These are figure-ready tables written by `figures/figure_utils.py`. Each row is one FFT size, and each plotted series contributes a `Mean` and `Std` column pair.

## Existing data in the repository

The repository already includes checked-in generated data, including:

- consolidated suite outputs under `test_results/`
- final figure assets under `figures/`

That means you can:

- inspect existing benchmark results directly from `test_results/`
- inspect existing figure-ready tables directly from `figures/*.csv`
- regenerate the figures from checked-in data without rerunning the benchmarks

## Cleaning generated outputs

To remove generated result directories before a fresh run:

```bash
bash clean_results.sh
```

This removes the top-level `test_results/` directory and the suite-local `tests/*/test_results/` directories listed in the script.
