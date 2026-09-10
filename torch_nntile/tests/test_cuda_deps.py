# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.

import importlib.util
import sys
from pathlib import Path

import pytest

_cuda_deps_path = (
    Path(__file__).resolve().parent.parent / "torch_nntile" / "_cuda_deps.py"
)
_spec = importlib.util.spec_from_file_location(
    "torch_nntile._cuda_deps",
    _cuda_deps_path,
)
assert _spec and _spec.loader
_cuda_deps = importlib.util.module_from_spec(_spec)
sys.modules["torch_nntile._cuda_deps"] = _cuda_deps
_spec.loader.exec_module(_cuda_deps)


def test_ensure_linux_cuda_deps_skips_when_not_required(monkeypatch):
    monkeypatch.setattr(_cuda_deps.sys, "platform", "linux")
    _cuda_deps.ensure_linux_cuda_deps(required=False)


def test_ensure_linux_cuda_deps_skips_non_linux(monkeypatch):
    monkeypatch.setattr(_cuda_deps.sys, "platform", "darwin")
    _cuda_deps.ensure_linux_cuda_deps(required=True)


def test_ensure_linux_cuda_deps_raises_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(_cuda_deps.sys, "platform", "linux")
    monkeypatch.setattr(_cuda_deps.sys, "path", [str(tmp_path)])
    monkeypatch.delenv("TORCH_LIB_DIR", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    with pytest.raises(ImportError, match="nvidia-cublas-cu12"):
        _cuda_deps.ensure_linux_cuda_deps(required=True)


def test_ensure_linux_cuda_deps_passes_when_present(monkeypatch, tmp_path):
    lib_root = tmp_path / "nvidia" / "cublas" / "lib"
    lib_root.mkdir(parents=True)
    (lib_root / "libcublas.so.12").write_bytes(b"")
    for folder in (
        "cudnn",
        "cusparse",
        "cusolver",
        "nvjitlink",
        "cuda_runtime",
    ):
        path = tmp_path / "nvidia" / folder / "lib"
        path.mkdir(parents=True)
        (path / "lib.so").write_bytes(b"")

    monkeypatch.setattr(_cuda_deps.sys, "platform", "linux")
    monkeypatch.setattr(_cuda_deps.sys, "path", [str(tmp_path)])

    _cuda_deps.ensure_linux_cuda_deps(required=True)


def test_ensure_linux_cuda_deps_passes_with_torch_lib_dir(monkeypatch, tmp_path):
    for _folder, candidates in _cuda_deps._LINUX_CUDA_SONAMES:
        (tmp_path / candidates[0]).write_bytes(b"")

    monkeypatch.setattr(_cuda_deps.sys, "platform", "linux")
    monkeypatch.setattr(_cuda_deps.sys, "path", [str(tmp_path)])
    monkeypatch.setenv("TORCH_LIB_DIR", str(tmp_path))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    _cuda_deps.ensure_linux_cuda_deps(required=True)
