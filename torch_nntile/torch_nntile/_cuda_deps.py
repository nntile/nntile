# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/_cuda_deps.py
# Linux CUDA wheel runtime dependency checks.

"""Verify NVIDIA CUDA 12 pip libraries before loading native extensions."""

from __future__ import annotations

import sys
from pathlib import Path

_LINUX_CUDA_NVIDIA_PACKAGES: tuple[tuple[str, str], ...] = (
    ("cublas", "nvidia-cublas-cu12"),
    ("cudnn", "nvidia-cudnn-cu12"),
    ("cusparse", "nvidia-cusparse-cu12"),
    ("cusolver", "nvidia-cusolver-cu12"),
    ("nvjitlink", "nvidia-nvjitlink-cu12"),
    ("cuda_runtime", "nvidia-cuda-runtime-cu12"),
)


def _nvidia_lib_dir(lib_folder: str) -> Path | None:
    for entry in sys.path:
        candidate = Path(entry) / "nvidia" / lib_folder / "lib"
        if candidate.is_dir() and any(candidate.glob("*.so*")):
            return candidate
    return None


def ensure_linux_cuda_deps() -> None:
    """Raise ImportError when required nvidia-*-cu12 packages are missing."""
    if sys.platform != "linux":
        return

    missing = [
        pip_name
        for lib_folder, pip_name in _LINUX_CUDA_NVIDIA_PACKAGES
        if _nvidia_lib_dir(lib_folder) is None
    ]
    if not missing:
        return

    packages = " ".join(missing)
    raise ImportError(
        "torch_nntile requires NVIDIA CUDA 12 libraries. Install: "
        f"pip install {packages} "
        "(or reinstall torch_nntile so pip resolves its dependencies)"
    )
