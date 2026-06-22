#!/usr/bin/env bash
# Install torch (cu128) and pip cuDNN; export build paths for libnntile/StarPU.
# Source this script; do not execute in a subshell.
#
# nvcc and cuda.h come from the system CUDA toolkit (install_linux_cuda_toolkit.sh).
# Pip nvidia-cuda-nvcc-cu12 does not ship nvcc; pip cudnn supplies cuDNN for cmake.
set -euo pipefail

torch_version="${TORCH_VERSION:-2.9.1}"
torch_cuda_index="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="$("${script_dir}/wheel_python.sh")"

"${python}" -m pip install --upgrade pip
"${python}" -m pip install numpy
# Index is cu128-only; pin torch==X.Y.Z without a +cu128 local tag.
"${python}" -m pip install \
    "torch==${torch_version}" \
    --index-url "${torch_cuda_index}"

"${python}" -m pip install nvidia-cudnn-cu12

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

export PATH="${CUDA_HOME}/bin:${PATH}"
export CMAKE_PREFIX_PATH="${TORCH_PREFIX}:${CUDA_HOME}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs:${TORCH_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cudnn_paths="$("${python}" - <<'PY'
from pathlib import Path
import importlib

mod = importlib.import_module("nvidia.cudnn")
if getattr(mod, "__file__", None):
    root = Path(mod.__file__).resolve().parent
elif paths := getattr(mod, "__path__", None):
    root = Path(next(iter(paths)))
else:
    raise RuntimeError("nvidia.cudnn is not installed or has no install path")

print(root)
print(root / "include")
print(root / "lib")
PY
)"
export CUDNN_PATH="$(echo "${cudnn_paths}" | sed -n '1p')"
export CUDNN_INCLUDE_PATH="$(echo "${cudnn_paths}" | sed -n '2p')"
export CUDNN_LIBRARY_PATH="$(echo "${cudnn_paths}" | sed -n '3p')"
