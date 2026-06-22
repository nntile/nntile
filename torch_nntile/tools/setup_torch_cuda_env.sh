#!/usr/bin/env bash
# Install torch from the PyTorch cu128 index and export CUDA paths for builds.
# Source this script; do not execute in a subshell.
set -euo pipefail

torch_version="${TORCH_VERSION:-2.9.1}"
torch_cuda_index="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="$("${script_dir}/wheel_python.sh")"

"${python}" -m pip install --upgrade pip
# Index is cu128-only; pin torch==X.Y.Z without a +cu128 local tag.
"${python}" -m pip install \
    "torch==${torch_version}" \
    --index-url "${torch_cuda_index}"

# nvcc is not bundled in the torch wheel.
"${python}" -m pip install nvidia-cuda-nvcc-cu12 nvidia-cudnn-cu12

export TORCH_PREFIX="$("${python}" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
export CUDA_HOME="$("${python}" - <<'PY'
from pathlib import Path
import nvidia.cuda_nvcc
print(Path(nvidia.cuda_nvcc.__file__).resolve().parent)
PY
)"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CMAKE_PREFIX_PATH="${TORCH_PREFIX}:${CUDA_HOME}"
export LD_LIBRARY_PATH="$("${python}" -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
