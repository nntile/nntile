# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/conftest.py
# Session-wide StarPU / nntile context for libnntile parity tests.

from __future__ import annotations

import os
import sys
from pathlib import Path

# Layout is torch_nntile/torch_nntile/. Prefer the in-tree package only when a
# built extension is present (editable / local build). Otherwise the project
# tree has no _C.so and must not shadow a pip-installed wheel (CI).
_pkg_root = Path(__file__).resolve().parent.parent
_pkg_dir = _pkg_root / "torch_nntile"
_repo_root = _pkg_root.parent
_has_local_ext = _pkg_root.name == "torch_nntile" and any(
    _pkg_dir.glob("_C.*")
)


def _path_entry(path: Path) -> str | None:
    try:
        return str(path.resolve())
    except OSError:
        return None


def _shadow_path_entries() -> set[str]:
    """Path entries that make the source tree win over a wheel install."""
    shadow: set[str] = set()
    root = _path_entry(_pkg_root)
    if root:
        shadow.add(root)
    repo = _path_entry(_repo_root)
    if repo and (_pkg_root / "pyproject.toml").is_file():
        shadow.add(repo)
    return shadow


if _has_local_ext:
    _root = _path_entry(_pkg_root)
    if _root and _root not in sys.path:
        sys.path.insert(0, _root)
else:
    _shadow = _shadow_path_entries()
    sys.path[:] = [
        p
        for p in sys.path
        if not p or _path_entry(Path(p)) not in _shadow
    ]

import pytest
import torch

import torch_nntile
from torch_nntile import _C


def subprocess_environ(**extra: str) -> dict[str, str]:
    """Env for child processes that ``import torch_nntile``.

    Prepends the in-tree project root to ``PYTHONPATH`` only when a local
    ``_C`` extension exists. Otherwise keeps site-packages (installed wheel)
    and strips source-tree entries that would shadow it.
    """
    env = dict(os.environ)
    env.pop("STARPU_DISABLE_KERNELS", None)
    env.pop("TORCH_NNTILE_SKIP_STARPU", None)

    build_lib = _repo_root / "build" / "nntile"
    starpu_lib = "/opt/starpu/lib"
    ld = env.get("LD_LIBRARY_PATH", "")
    for part in (str(build_lib), starpu_lib):
        if part and part not in ld.split(":"):
            ld = f"{part}:{ld}" if ld else part
    # Installed libnntile / libtorch_nntile (CI prefix jobs).
    prefix_lib = os.environ.get("NNTILE_PREFIX", "")
    if prefix_lib:
        candidate = str(Path(prefix_lib) / "lib")
        if candidate not in ld.split(":"):
            ld = f"{candidate}:{ld}" if ld else candidate
    env["LD_LIBRARY_PATH"] = ld

    pkg_root = _path_entry(_pkg_root) or str(_pkg_root)
    existing = [p for p in env.get("PYTHONPATH", "").split(":") if p]
    if _has_local_ext:
        ordered = [pkg_root] + [p for p in existing if p != pkg_root]
    else:
        shadow = _shadow_path_entries()
        ordered = [
            p for p in existing if _path_entry(Path(p)) not in shadow
        ]
    env["PYTHONPATH"] = ":".join(ordered)
    env.update(extra)
    return env


def ensure_nntile_context(
    *,
    ncpu: int = 1,
    ncuda: int = 0,
    verbose: int = 0,
    cpu_fallback: bool = False,
) -> None:
    """Initialize the nntile context before the first libnntile-backed op.

    ``cpu_fallback`` is selected at :func:`init_context` time (runtime flag, not
    a compile-time setting). Call this early with ``cpu_fallback=False`` for
    parity tests that require unsupported ATen ops to fail instead of silently
    falling back to CPU.
    """
    if not _C.has_libnntile():
        return
    if not torch_nntile.is_context_initialized():
        torch_nntile.init_context(
            ncpu=ncpu,
            ncuda=ncuda,
            verbose=verbose,
            cpu_fallback=cpu_fallback,
        )
        torch_nntile.restrict_cpu()
        return
    if cpu_fallback and torch_nntile.is_cpu_fallback_enabled():
        return
    if not cpu_fallback and torch_nntile.is_cpu_fallback_enabled():
        pytest.fail(
            "nntile context already initialized with cpu_fallback=True; "
            "tests in this process require cpu_fallback=False"
        )
    torch_nntile.restrict_cpu()


def pytest_sessionstart(session) -> None:
    """Configure nntile before collection or any libnntile-backed op."""
    del session
    ensure_nntile_context(cpu_fallback=False)


def pytest_sessionfinish(session, exitstatus) -> None:
    """Tear down StarPU before interpreter finalization (avoids exit UAF)."""
    del session, exitstatus
    if not _C.has_libnntile():
        return
    if torch_nntile.is_context_initialized():
        torch_nntile.wait()
        torch_nntile.shutdown_context()


@pytest.fixture(autouse=True)
def _reset_nntile_graph_session_after_test():
    """Isolate parity tests: stale TensorGraph sessions corrupt later tests."""
    yield
    if _C.has_libnntile():
        torch_nntile.reset_graph_session()


def nntile_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Copy an nntile tensor to CPU, flushing a pending TensorGraph first."""
    if (
        _C.has_libnntile()
        and tensor.device.type == "nntile"
        and torch_nntile.has_pending_graph()
    ):
        torch_nntile.compile_graph()
        torch_nntile.run()
    with torch.no_grad():
        return tensor.cpu()
