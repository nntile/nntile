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
# Fresh cibuildwheel test venvs reinstall ~2.5GB CUDA torch while the
# build tree is still on disk; reclaim cache/tmp before the download.
"${python}" -m pip cache purge 2>/dev/null || true
rm -rf /root/.cache/pip "${TMPDIR:-/tmp}/pip-"* 2>/dev/null || true
# Drop unrepaired / intermediate wheel trees if present (repair already
# wrote the final artifact). Safe no-ops outside cibuildwheel.
rm -rf /tmp/cibuildwheel/built_wheel /tmp/cibuildwheel/repaired_wheel \
    2>/dev/null || true
# Shared libs are already linked into the repaired wheel; the cmake tree
# is several GB and is the usual disk-full culprit on GHA runners.
if [ -n "${NNTILE_BUILD_DIR:-}" ]; then
    rm -rf "${NNTILE_BUILD_DIR}" 2>/dev/null || true
fi
rm -rf /tmp/nntile-build 2>/dev/null || true
df -h / /tmp 2>/dev/null || df -h / || true
"${python}" -m pip install --no-cache-dir \
    "torch==${torch_version}" \
    "torchvision==${torchvision_version}" \
    --index-url "${torch_cuda_index}"
"${python}" -m pip cache purge 2>/dev/null || true
