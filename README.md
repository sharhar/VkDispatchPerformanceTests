# VkDispatchPerformanceTests

This repository contains benchmark and accuracy tests for `vkdispatch`. It is intended to evaluate `vkdispatch` itself for paper-quality comparisons and repeatable measurements.

The primary user-facing entrypoint for running benchmarks is `run_all_tests.py`.

## Dependencies

Required tools: Python 3, `git`, `wget`, `tar`, and `bash`

Required Python packages: `numpy`, and `matplotlib`

### vkdispatch and backends

`vkdispatch` can be [built from source](https://sharhar.github.io/vkdispatch/tutorials/building_from_source.html) or installed from PyPI:

```bash
pip install vkdispatch
```

This package includes the core `vkdispatch` library and the Vulkan backend. The OpenCL and CUDA backends can be optionally enabled by installing the `pyopencl` and `cuda-python` packages, respectively.

### CUDA Benchmarks

CUDA benchmark helper binaries and cuFFT/cuFFTDx comparisons require a CUDA toolkit (version 12 or higher) installation with `nvcc`. `run_all_tests.py` checks `CUDA_HOME/bin/nvcc` first if `CUDA_HOME` is set, otherwise it uses `nvcc` from `PATH`

CUDA accuracy helpers and cuFFTDx benchmarks rely on downloaded NVIDIA dependencies in `dependencies/`. These are automatically downloaded on the first run of `run_all_tests.py` and pinned to specific revisions for repeatability:

- [NVIDIA MathDx version 25.06.1](https://developer.nvidia.com/downloads/compute/cuFFTDx/redist/cuFFTDx/cuda12/nvidia-mathdx-25.06.1-cuda12.tar.gz)
- [cutlass](https://github.com/NVIDIA/cutlass.git) pinned to commit `e6e2cc29f5e7611dfc6af0ed6409209df0068cf2`
- [CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples.git) pinned to commit `a94482ebecf8b16d5b83ab276b7db3a84979f0e5` (used by the `tests/conv_scaled_nvidia/` test suite)

## Running the tests

`run_all_tests.py` is the main entrypoint for benchmark and accuracy generation. It accepts flags to specify which backends to test and whether to run validation-only paths for CUDA benchmarks. Some example invocations:

```bash
# For all 3 backends
python3 run_all_tests.py --vulkan --opencl --cuda

# For Vulkan only (this is the only backend that includes VkFFT)
python3 run_all_tests.py --vulkan

# For OpenCL only
python3 run_all_tests.py --opencl

# For CUDA only, with normal benchmarks
python3 run_all_tests.py --cuda
```

There is also a validation-only path for CUDA benchmarks that compares the cufftDx fused convolution outputs against the standard cufft + pointwise reference instead of running the normal throughput benchmarks. To run this path:

```bash
python3 run_all_tests.py --validate --cuda
```
Current built-in benchmark settings in `run_all_tests.py`:

```python
DATA_SIZE = 2**27
ITER_COUNT = 200
BATCH_SIZE = 20
REPEATS = 5
```

Notes about first-time setup:
- Raw benchmark outputs are first written into `tests/<suite>/test_results/`, then copied and merged into the top-level `test_results/` tree
- `test_results/` is created locally when the benchmarks run; it does not exist in the public repository until you generate results

## Cleaning generated outputs

To remove generated result directories before a fresh run:

```bash
bash clean_results.sh
```

This removes the top-level `test_results/` directory and the suite-local `tests/*/test_results/` directories listed in the script.
