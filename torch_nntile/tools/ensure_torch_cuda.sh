#!/usr/bin/env bash
# Ensure torch+cu128 is importable in the current Python (cibuildwheel hooks).
# Skips reinstall when before-all already installed the same stack into this
# interpreter — avoids a second multi-GB download on the runner disk.
set -euo pipefail

torch_version="${TORCH_VERSION:-2.9.1}"
torchvision_version="${TORCHVISION_VERSION:-0.24.1}"
torch_cuda_index="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
python="${WHEEL_PYTHON:-python}"

if "${python}" - <<PY
import sys
try:
    import torch
except ImportError:
    sys.exit(1)
ver = torch.__version__.split("+", 1)[0]
if ver != "${torch_version}":
    sys.exit(1)
if not torch.version.cuda:
    sys.exit(1)
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) already present")
PY
then
    exit 0
fi

echo "Installing torch==${torch_version} from ${torch_cuda_index}" >&2
"${python}" -m pip install --no-cache-dir \
    "torch==${torch_version}" \
    "torchvision==${torchvision_version}" \
    --index-url "${torch_cuda_index}"
