import sys
import csv
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import importlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import run_accuracy_test, AccuracyConfig, accuracy_helpers

import vkdispatch as vd
import numpy as np


#accuracy_helpers = importlib.import_module("common.run_accuracy_test")


def vkdispatch_fft_test_function(config: AccuracyConfig,
                                 fft_size: int,
                                 buffer: vd.Buffer,
                                 kernel: vd.Buffer):
    vd.fft.fft(buffer)


def vkfft_test_function(config: AccuracyConfig,
                        fft_size: int,
                        buffer: vd.Buffer,
                        kernel: vd.Buffer):
    vd.vkfft.fft(buffer)


def _find_nvcc() -> Optional[str]:
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        candidate = Path(cuda_home) / "bin" / "nvcc"
        if candidate.exists():
            return str(candidate)

    return shutil.which("nvcc")


def _detect_cuda_arch(nvcc_path: str, root_dir: Path, build_dir: Path) -> Optional[int]:
    arch_source = root_dir / "arch_code.cu"
    arch_exec = build_dir / "arch_code_accuracy.exec"

    try:
        subprocess.run(
            [nvcc_path, str(arch_source), "-o", str(arch_exec)],
            cwd=build_dir,
            check=True,
            capture_output=True,
            text=True
        )
        result = subprocess.run(
            [str(arch_exec)],
            cwd=build_dir,
            check=True,
            capture_output=True,
            text=True
        )
        return int(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as error:
        if isinstance(error, subprocess.CalledProcessError):
            details = (error.stderr or error.stdout or str(error)).strip()
        else:
            details = str(error)
        print(f"Skipping CUDA accuracy tests: failed to detect CUDA architecture ({details})")
        return None
    finally:
        if arch_exec.exists():
            arch_exec.unlink()


def _compile_cuda_accuracy_binary() -> Optional[Path]:
    if platform.system() == "Darwin":
        print("Skipping CUDA accuracy tests on macOS")
        return None

    nvcc_path = _find_nvcc()
    if nvcc_path is None:
        print("Skipping CUDA accuracy tests: nvcc not found")
        return None

    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parents[1]
    build_dir = Path.cwd()

    source_file = script_dir / "cuda_accuracy_test.cu"
    if not source_file.exists():
        print(f"Skipping CUDA accuracy tests: source not found at {source_file}")
        return None

    cutlass_include = root_dir / "dependencies" / "cutlass" / "include"
    mathdx_include = (
        root_dir
        / "dependencies"
        / "nvidia-mathdx-25.06.1"
        / "nvidia"
        / "mathdx"
        / "25.06"
        / "include"
    )

    if not cutlass_include.exists() or not mathdx_include.exists():
        print("Skipping CUDA accuracy tests: CUDA dependencies are missing. Run run_all_tests.py once to fetch them.")
        return None

    cuda_arch = _detect_cuda_arch(nvcc_path, root_dir, build_dir)
    if cuda_arch is None:
        return None

    executable = build_dir / f"cuda_accuracy_test_sm{cuda_arch}.exec"
    needs_rebuild = (not executable.exists()) or (source_file.stat().st_mtime > executable.stat().st_mtime)

    if not needs_rebuild:
        return executable

    compile_cmd = [
        nvcc_path,
        str(source_file),
        "-std=c++17",
        "-O3",
        "-I",
        str(cutlass_include),
        "-I",
        str(mathdx_include),
        f"-DARCH={cuda_arch}0",
        "-gencode",
        f"arch=compute_{cuda_arch},code=sm_{cuda_arch}",
        "-lcufft",
        "-lculibos",
        "-o",
        str(executable),
    ]

    try:
        subprocess.run(
            compile_cmd,
            cwd=build_dir,
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or str(error)).strip()
        print(f"Skipping CUDA accuracy tests: failed to compile helper executable ({details})")
        return None

    return executable


def _run_cuda_backend_once(config: AccuracyConfig,
                           fft_size: int,
                           run_index: int,
                           backend_name: str,
                           executable: Path,
                           temp_dir: Path):
    input_data = config.make_random_data(fft_size, run_index)
    reference = np.fft.fft(input_data)

    input_flat = np.ascontiguousarray(input_data.reshape(-1), dtype=np.complex64)
    input_file = temp_dir / f"{backend_name}_input.bin"
    output_file = temp_dir / f"{backend_name}_output.bin"
    input_flat.tofile(input_file)

    cmd = [
        str(executable),
        backend_name,
        str(fft_size),
        str(config.data_size),
        str(input_file),
        str(output_file),
    ]

    try:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            details = (error.stderr or error.stdout or str(error)).strip()
            raise RuntimeError(
                f"CUDA backend '{backend_name}' failed for FFT size {fft_size}, run {run_index}: {details}"
            ) from error

        result_flat = np.fromfile(output_file, dtype=np.complex64)
        if result_flat.size != config.data_size:
            raise RuntimeError(
                f"Unexpected output size for backend '{backend_name}': "
                f"expected {config.data_size}, got {result_flat.size}"
            )
    finally:
        if input_file.exists():
            input_file.unlink()
        if output_file.exists():
            output_file.unlink()

    result_data = np.ascontiguousarray(result_flat.reshape(config.make_shape(fft_size)))
    return accuracy_helpers._compute_metrics(reference, result_data)


def _run_cuda_accuracy_test(output_name: str,
                            backend_name: str,
                            executable: Path):
    config = accuracy_helpers.parse_args()
    fft_sizes = accuracy_helpers.get_fft_sizes()

    with open(f"{output_name}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["Backend", "FFT Size"]
            + [f"Run {i + 1} (Relative L2 Error)" for i in range(config.run_count)]
            + ["Mean", "Std Dev", "Worst Max Relative Error", "Worst Max Absolute Error"]
        )

        with tempfile.TemporaryDirectory(prefix=f"{backend_name}_accuracy_") as temp_path:
            temp_dir = Path(temp_path)

            for fft_size in fft_sizes:
                rel_l2_errors = []
                max_relative_errors = []
                max_absolute_errors = []

                for run_index in range(config.run_count):
                    relative_l2, max_relative, max_absolute = _run_cuda_backend_once(
                        config,
                        fft_size,
                        run_index,
                        backend_name,
                        executable,
                        temp_dir
                    )
                    rel_l2_errors.append(relative_l2)
                    max_relative_errors.append(max_relative)
                    max_absolute_errors.append(max_absolute)

                    print(
                        f"[{backend_name}] FFT Size: {fft_size}, "
                        f"Relative L2 Error: {relative_l2:.6e}, "
                        f"Max Relative Error: {max_relative:.6e}, "
                        f"Max Absolute Error: {max_absolute:.6e}"
                    )

                writer.writerow(
                    [backend_name, fft_size]
                    + [f"{value:.6e}" for value in rel_l2_errors]
                    + [
                        f"{np.mean(rel_l2_errors):.6e}",
                        f"{np.std(rel_l2_errors):.6e}",
                        f"{np.max(max_relative_errors):.6e}",
                        f"{np.max(max_absolute_errors):.6e}",
                    ]
                )

    print(f"Accuracy results saved to {output_name}.csv")


def run_cuda_accuracy_tests_if_available():
    executable = _compile_cuda_accuracy_binary()
    if executable is None:
        return

    _run_cuda_accuracy_test("cufft_accuracy", "cufft", executable)
    _run_cuda_accuracy_test("cufftdx_accuracy", "cufftdx", executable)


if __name__ == "__main__":
    run_accuracy_test("vkdispatch_accuracy", "vkdispatch", vkdispatch_fft_test_function)
    run_accuracy_test("vkfft_accuracy", "vkfft", vkfft_test_function)
    run_cuda_accuracy_tests_if_available()
