# VkDispatchPerformanceTests

This repository contains benchmark and accuracy tests for `vkdispatch`, plus scripts for turning benchmark outputs into publication-style figures. It is intended to evaluate `vkdispatch` itself for paper-quality comparisons and repeatable measurements.

The primary user-facing entrypoint for running benchmarks is `run_all_tests.py`. Figure generation is handled separately by the scripts under `figures/`.

## Repository layout

- `run_all_tests.py`: main benchmark and accuracy runner
- `tests/`: per-suite source code and raw per-run output directories
- `figures/`: scripts for final figure generation
- `test_results/`: created when benchmarks are run; holds consolidated benchmark and accuracy CSVs plus quick-look graphs
- `dependencies/`: created when the test scripts run; holds downloaded third-party CUDA dependencies

In a fresh public clone, `dependencies/`, `test_results/`, generated CSVs, generated PNGs, and generated PDFs are not present. They are gitignored and created locally when you run the code.

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

Install `vkdispatch` from PyPI:

```bash
pip install vkdispatch
```

On mainstream platforms, `pip install vkdispatch` will install the Python package together with a precompiled native Vulkan backend. If no compatible precompiled wheel is available for your platform, `pip` falls back to building from source, which requires a C++17 compiler. Additional vkdispatch build instructions are available in the vkdispatch docs: [Building From Source](https://sharhar.github.io/vkdispatch/tutorials/building_from_source.html).

Backend-specific requirements:

- Vulkan benchmarks require a working Vulkan runtime and driver stack
- OpenCL benchmarks require `pyopencl` plus a working OpenCL runtime and driver stack
- CUDA `vkdispatch` benchmarks require NVIDIA's `cuda-python` package
- CUDA benchmark helper binaries and cuFFT/cuFFTDx comparisons require a CUDA toolkit installation with `nvcc`
- `run_all_tests.py` checks `CUDA_HOME/bin/nvcc` first if `CUDA_HOME` is set, otherwise it uses `nvcc` from `PATH`

Backend package installation examples:

```bash
pip install pyopencl
pip install cuda-python
```

CUDA accuracy helpers and cuFFTDx benchmarks rely on downloaded NVIDIA dependencies in `dependencies/`.

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
- The NVIDIA convolution helper under `tests/conv_scaled_nvidia/` also creates and uses `dependencies/CUDALibrarySamples`, pinned to commit `a94482ebecf8b16d5b83ab276b7db3a84979f0e5`
- These pinned dependency revisions are used to improve repeatability for paper results
- Raw benchmark outputs are first written into `tests/<suite>/test_results/`, then copied and merged into the top-level `test_results/` tree
- `test_results/` is created locally when the benchmarks run; it does not exist in the public repository until you generate results

## Making the figures

The final figures are generated from the consolidated root-level data under `test_results/`.

Run:

```bash
bash make_figures.sh
```

Figure outputs:

- `figures/*.pdf`
- `figures/*.png`
- `figures/*.csv`

These files are generated locally and are gitignored in the public repository. In a fresh clone, you must run the benchmarks first so that `test_results/` exists before generating `fig1` through `fig4`.

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

`test_results/` is created on demand by the benchmark scripts and is not included in the public repo.

### 3. Final figure assets

The scripts under `figures/` read from `../test_results/...` and write final publication-style assets into:

- `figures/*.pdf`
- `figures/*.png`
- `figures/*.csv`

These figure outputs are also generated locally and are not included in the public repo.

## CSV formats

### Benchmark backend CSVs

Example generated paths:

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

Example generated paths:

- `test_results/conv_scaled_control/merged.csv`
- `test_results/accuracy/merged.csv`

Format:

```text
Backend,FFT Size,Mean,Std Dev
```

These files collapse the per-backend CSVs into one table per suite for plotting and inspection.

### Accuracy CSVs

Example generated paths:

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

Example generated paths:

- `figures/fig1_scaled_nonstrided_convolution.csv`
- `figures/fig4_accuracy.csv`

Format:

```text
FFT Size,<Series 1> Mean,<Series 1> Std,<Series 2> Mean,<Series 2> Std,...
```

These are figure-ready tables written by `figures/figure_utils.py`. Each row is one FFT size, and each plotted series contributes a `Mean` and `Std` column pair.

## Generated artifacts and gitignore

The public repository intentionally does not include generated benchmark data or rendered figures. The root `.gitignore` excludes:

- `dependencies/`
- `*.csv`
- `*.png`
- `*.pdf`

As a result:

- `dependencies/` appears only after the scripts download third-party code
- `test_results/` appears only after benchmarks or accuracy runs complete
- rendered figure files under `figures/` appear only after `bash make_figures.sh`

If you want figures, you need to generate the data first and then run the figure scripts.

## Cleaning generated outputs

To remove generated result directories before a fresh run:

```bash
bash clean_results.sh
```

This removes the top-level `test_results/` directory and the suite-local `tests/*/test_results/` directories listed in the script.
