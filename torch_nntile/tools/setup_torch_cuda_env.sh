#!/usr/bin/env bash
# Install torch (cu128) once and export pip nvidia paths for libnntile/StarPU.
# Source this script; do not execute in a subshell.
#
# nvcc and libcuda stubs come from the thin system toolkit
# (install_linux_cuda_toolkit.sh). Math/runtime libs (cudart, cublas, cudnn)
# come from pip nvidia-*-cu12 packages pulled in by torch — not from
# /usr/local/cuda/lib64.
set -euo pipefail

torch_version="${TORCH_VERSION:-2.9.1}"
torch_cuda_index="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="$("${script_dir}/wheel_python.sh")"

disk_log() {
    if [ "${TORCH_NNTILE_DISK_LOG:-0}" != "1" ]; then
        return 0
    fi
    echo "[disk] $*:" >&2
    df -h / /tmp 2>/dev/null || df -h /
}

disk_log "before pip torch"
"${python}" -m pip install --upgrade pip
"${python}" -m pip install --no-cache-dir numpy
# Index is cu128-only; pin torch==X.Y.Z without a +cu128 local tag.
# Transitive deps already include nvidia-{cuda_runtime,cublas,cudnn,...}-cu12.
"${python}" -m pip install --no-cache-dir \
    "torch==${torch_version}" \
    "torchvision==0.24.1" \
    --index-url "${torch_cuda_index}"
"${python}" -m pip cache purge || true
disk_log "after pip torch"

export TORCH_PREFIX="$("${python}" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
export TORCH_LIB_DIR="$("${python}" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"

if [ -z "${CUDA_HOME:-}" ]; then
    if [ -d /usr/local/cuda ]; then
        export CUDA_HOME=/usr/local/cuda
    else
        echo "CUDA_HOME is unset and /usr/local/cuda is missing; run install_linux_cuda_toolkit.sh" >&2
        exit 1
    fi
fi

# Resolve pip nvidia package roots (include + lib) for cmake / LD_LIBRARY_PATH.
# StarPU's CUDA headers pull in cusolver/cusparse even when those libs are not
# linked; expose their pip include dirs for the StarPU build.
nvidia_env="$("${python}" - <<'PY'
from pathlib import Path
import importlib

def pkg_paths(modname: str) -> tuple[str, str, str]:
    mod = importlib.import_module(modname)
    if getattr(mod, "__file__", None):
        root = Path(mod.__file__).resolve().parent
    elif paths := getattr(mod, "__path__", None):
        root = Path(next(iter(paths))).resolve()
    else:
        raise RuntimeError(f"{modname} has no install path")
    include = root / "include"
    lib = root / "lib"
    if not include.is_dir():
        raise RuntimeError(f"missing include dir for {modname}: {include}")
    if not lib.is_dir():
        raise RuntimeError(f"missing lib dir for {modname}: {lib}")
    return str(root), str(include), str(lib)

for key, modname in (
    ("CUDNN", "nvidia.cudnn"),
    ("NVIDIA_CUBLAS", "nvidia.cublas"),
    ("NVIDIA_CUDA_RUNTIME", "nvidia.cuda_runtime"),
    ("NVIDIA_CUSOLVER", "nvidia.cusolver"),
    ("NVIDIA_CUSPARSE", "nvidia.cusparse"),
):
    root, include, lib = pkg_paths(modname)
    print(f"{key}_PATH={root}")
    print(f"{key}_INCLUDE_PATH={include}")
    print(f"{key}_LIBRARY_PATH={lib}")
PY
)"
eval "${nvidia_env}"
export CUDNN_PATH CUDNN_INCLUDE_PATH CUDNN_LIBRARY_PATH
export NVIDIA_CUBLAS_PATH NVIDIA_CUBLAS_INCLUDE_PATH NVIDIA_CUBLAS_LIBRARY_PATH
export NVIDIA_CUDA_RUNTIME_PATH NVIDIA_CUDA_RUNTIME_INCLUDE_PATH
export NVIDIA_CUDA_RUNTIME_LIBRARY_PATH
export NVIDIA_CUSOLVER_PATH NVIDIA_CUSOLVER_INCLUDE_PATH
export NVIDIA_CUSOLVER_LIBRARY_PATH
export NVIDIA_CUSPARSE_PATH NVIDIA_CUSPARSE_INCLUDE_PATH
export NVIDIA_CUSPARSE_LIBRARY_PATH
export NNTILE_CUDA_FROM_PIP=1

# Prefer pip math/runtime libs over any residual toolkit copies.
pip_lib_path="${NVIDIA_CUBLAS_LIBRARY_PATH}:${CUDNN_LIBRARY_PATH}:${NVIDIA_CUDA_RUNTIME_LIBRARY_PATH}:${NVIDIA_CUSOLVER_LIBRARY_PATH}:${NVIDIA_CUSPARSE_LIBRARY_PATH}"
pip_inc_path="${NVIDIA_CUBLAS_INCLUDE_PATH}:${CUDNN_INCLUDE_PATH}:${NVIDIA_CUDA_RUNTIME_INCLUDE_PATH}:${NVIDIA_CUSOLVER_INCLUDE_PATH}:${NVIDIA_CUSPARSE_INCLUDE_PATH}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CMAKE_PREFIX_PATH="${TORCH_PREFIX}:${CUDA_HOME}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export CMAKE_LIBRARY_PATH="${pip_lib_path}${CMAKE_LIBRARY_PATH:+:${CMAKE_LIBRARY_PATH}}"
export CMAKE_INCLUDE_PATH="${pip_inc_path}${CMAKE_INCLUDE_PATH:+:${CMAKE_INCLUDE_PATH}}"
export CPATH="${pip_inc_path}${CPATH:+:${CPATH}}"
export C_INCLUDE_PATH="${pip_inc_path}${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
export CPLUS_INCLUDE_PATH="${pip_inc_path}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export LD_LIBRARY_PATH="${pip_lib_path}:${CUDA_HOME}/lib64/stubs:${TORCH_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# Toolkit lib64 last (stubs + any residual cudart); never ahead of pip nvidia.
if [ -d "${CUDA_HOME}/lib64" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"
fi
