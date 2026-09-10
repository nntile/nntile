# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/_cuda_deps.py
# Linux CUDA runtime dependency checks.

"""Verify NVIDIA CUDA 12 libraries before loading native extensions.

Accepts pip ``nvidia-*-cu12`` layout or CUDA sonames under ``TORCH_LIB_DIR``
(or ``torch/lib``), ``CONDA_PREFIX/lib``, ``CUDA_HOME``, ``LD_LIBRARY_PATH``.
"""

from __future__ import annotations

import os
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

_LINUX_CUDA_SONAMES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cublas", ("libcublas.so.12",)),
    ("cudnn", ("libcudnn.so.9",)),
    ("cusparse", ("libcusparse.so.12",)),
    ("cusolver", ("libcusolver.so.12", "libcusolver.so.11")),
    ("nvjitlink", ("libnvjitlink.so.12", "libnvJitLink.so.12")),
    ("cuda_runtime", ("libcudart.so.12",)),
)


def _nvidia_lib_dir(lib_folder: str) -> Path | None:
    for entry in sys.path:
        candidate = Path(entry) / "nvidia" / lib_folder / "lib"
        if candidate.is_dir() and any(candidate.glob("*.so*")):
            return candidate
    return None


def _pip_nvidia_layout_ok() -> bool:
    return all(
        _nvidia_lib_dir(lib_folder) is not None
        for lib_folder, _pip_name in _LINUX_CUDA_NVIDIA_PACKAGES
    )


def _torch_lib_dir() -> Path | None:
    env = os.environ.get("TORCH_LIB_DIR")
    if env:
        path = Path(env)
        if path.is_dir():
            return path
    try:
        import torch
    except ImportError:
        return None
    path = Path(torch.__file__).resolve().parent / "lib"
    return path if path.is_dir() else None


def _cuda_lib_search_dirs() -> list[Path]:
    dirs: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path | None) -> None:
        if path is None or not path.is_dir():
            return
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        dirs.append(resolved)

    add(_torch_lib_dir())
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        add(Path(prefix) / "lib")
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        add(Path(cuda_home) / "lib64")
        add(Path(cuda_home) / "lib")
    for part in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if part:
            add(Path(part))
    return dirs


def _soname_visible(soname: str, search_dirs: list[Path]) -> bool:
    for directory in search_dirs:
        if (directory / soname).is_file():
            return True
    return False


def _any_soname_visible(candidates: tuple[str, ...], search_dirs: list[Path]) -> bool:
    return any(_soname_visible(soname, search_dirs) for soname in candidates)


def _torch_or_toolkit_layout_ok() -> bool:
    search_dirs = _cuda_lib_search_dirs()
    if not search_dirs:
        return False
    return all(
        _any_soname_visible(candidates, search_dirs)
        for _folder, candidates in _LINUX_CUDA_SONAMES
    )


def ensure_linux_cuda_deps(*, required: bool | None = None) -> None:
    """Raise ImportError when CUDA libraries are not discoverable.

    CPU-only builds skip this check. Pass ``required=True/False`` to
    override; ``None`` reads ``_build_info.BUILT_WITH_CUDA``.
    """
    if required is None:
        from ._build_info import BUILT_WITH_CUDA

        required = bool(BUILT_WITH_CUDA)
    if not required or sys.platform != "linux":
        return

    if _pip_nvidia_layout_ok() or _torch_or_toolkit_layout_ok():
        return

    packages = " ".join(pip_name for _folder, pip_name in _LINUX_CUDA_NVIDIA_PACKAGES)
    raise ImportError(
        "torch_nntile was built with CUDA and requires NVIDIA CUDA 12 "
        f"libraries. Install: pip install {packages} "
        "(or set TORCH_LIB_DIR / CONDA_PREFIX / CUDA_HOME / LD_LIBRARY_PATH "
        "so libcublas, libcudnn, libcusparse, libcusolver, libnvjitlink, "
        "and libcudart are visible — see torch_nntile/tools/setup_torch_cuda_env.sh)"
    )
