#!/usr/bin/env bash
# Install torch 2.9.1+cu128 and export CUDA paths for libstarpu/libnntile builds.
# Source this script; do not execute in a subshell.
set -euo pipefail

torch_version="${TORCH_VERSION:-2.9.1}"
torch_cuda_tag="${TORCH_CUDA_TAG:-cu128}"
torch_cuda_index="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"

python -m pip install --upgrade pip
python -m pip install \
    "torch==${torch_version}+${torch_cuda_tag}" \
    --index-url "${torch_cuda_index}"

# nvcc is not bundled in the torch wheel.
python -m pip install nvidia-cuda-nvcc-cu12 nvidia-cudnn-cu12

export TORCH_PREFIX="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
export CUDA_HOME="$(python - <<'PY'
from pathlib import Path
import nvidia.cuda_nvcc
print(Path(nvidia.cuda_nvcc.__file__).resolve().parent)
PY
)"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CMAKE_PREFIX_PATH="${TORCH_PREFIX}:${CUDA_HOME}"
export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
