# Data format and repository layout

## Repository layout

- `run_all_tests.py`: main benchmark and accuracy runner
- `tests/`: per-suite source code and raw per-run output directories
- `figures/`: scripts for final figure generation
- `test_results/`: created when benchmarks are run; holds consolidated benchmark and accuracy CSVs plus quick-look graphs
- `dependencies/`: created when the test scripts run; holds downloaded third-party CUDA dependencies

In a fresh public clone, `dependencies/`, `test_results/`, generated CSVs, generated PNGs, and generated PDFs are not present. They are gitignored and created locally when you run the code.


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
