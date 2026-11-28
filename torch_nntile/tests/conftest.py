# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/conftest.py
# Session-wide NNHaul / nntile context for device="nntile" parity tests.
# @version 2.0.0

from __future__ import annotations

import os
import sys
from pathlib import Path

# TensorGraph lowering is opt-in in the product; the test session always runs
# with the local compiler so a bare ``pytest`` exercises the real graph path.
os.environ.setdefault("NNTILE_ENABLE_LOCAL_COMPILER", "1")

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
        p for p in sys.path if not p or _path_entry(Path(p)) not in _shadow
    ]

import pytest  # noqa: E402
import torch  # noqa: E402

import torch_nntile  # noqa: E402


def subprocess_environ(**extra: str) -> dict[str, str]:
    """Env for child processes that ``import torch_nntile``.

    Prepends the in-tree project root to ``PYTHONPATH`` only when a local
    ``_C`` extension exists. Otherwise keeps site-packages (installed wheel)
    and strips source-tree entries that would shadow it.
    """
    env = dict(os.environ)
    env.pop("TORCH_NNTILE_SKIP_SUBMIT", None)
    env.pop("TORCH_NNTILE_SKIP_KERNELS", None)
    # Child contexts size their own arena; inherited caps can be zero for
    # the child's device and must not leak between sessions.
    env.pop("NNHAUL_LIMIT_CUDA_MEM", None)
    env.pop("NNHAUL_LIMIT_CPU_MEM", None)
    env.pop("NNHAUL_CHUNK_SIZE_MB", None)
    # A child context shares the GPU with this pytest process; give it a
    # fixed cap so auto-sizing cannot compute zero from the free memory
    # the parent context holds.
    env["NNHAUL_LIMIT_CUDA_MEM"] = "512"
    env["NNHAUL_WORKSPACE_MB"] = "128"
    env["NNHAUL_LIMIT_CPU_MEM"] = "256"
    # Child ``python -c`` must not put the source tree's package first:
    # only the installed wheel carries the compiled _C extension.
    env["PYTHONSAFEPATH"] = "1"

    build_root = Path(
        os.environ.get("NNTILE_BUILD_DIR") or _repo_root / "build-nnhaul"
    )
    if not (build_root / "nntile").is_dir():
        build_root = _repo_root / "build"
    ld = env.get("LD_LIBRARY_PATH", "")
    # Build-tree libraries are appended; the installed prefix (CI prefix jobs)
    # stays ahead of them so a leftover build tree cannot shadow it.
    for part in (str(build_root / "nntile"), str(build_root / "nnhaul")):
        if part and part not in ld.split(":"):
            ld = f"{ld}:{part}" if ld else part
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
        ordered = [p for p in existing if _path_entry(Path(p)) not in shadow]
    env["PYTHONPATH"] = ":".join(ordered)
    env.update(extra)
    return env


def ensure_nntile_context(
    *,
    ncpu: int = 0,
    ncuda: int = 1,
    verbose: int = 0,
    cpu_fallback: bool = False,
) -> None:
    """Initialize the NNHaul CUDA context before the first nntile op."""
    if not torch_nntile.is_context_initialized():
        torch_nntile.init_context(
            ncpu=ncpu,
            ncuda=ncuda,
            verbose=verbose,
            cpu_fallback=cpu_fallback,
        )
    if cpu_fallback and torch_nntile.is_cpu_fallback_enabled():
        return
    if not cpu_fallback and torch_nntile.is_cpu_fallback_enabled():
        pytest.fail(
            "nntile context already initialized with cpu_fallback=True; "
            "tests in this process require cpu_fallback=False"
        )

def pytest_collection_modifyitems(config, items) -> None:
    """Skip classic-kernel tests only when pybind lacks classic symbols."""
    del config
    import torch_nntile

    try:
        _ = torch_nntile._C.gemm
    except AttributeError:
        skip = pytest.mark.skip(
            reason="Classic NNTile kernels not linked in this build"
        )
        classic_only = {
            "test_add_fiber_parity",
            "test_transformer_layer_parity",
            "test_cross_entropy_parity",
            "test_sgd_step_parity",
            "test_adam_step_parity",
            "test_deep_relu_parity",
            "test_cpp_models_smoke",
            "test_simple_matmul_tiling",
            "test_bmm_tiling",
            "test_gpt2_lm_head_parity",
            "test_mlp_mixer_parity",
            "test_nn_classic",
            "test_models_classic_graph",
            "test_models_python_classic_graph",
            "test_models_python_tiled_smoke",
        }
        for item in items:
            mod = item.module.__name__.rsplit(".", 1)[-1]
            if mod in classic_only:
                item.add_marker(skip)


def pytest_sessionstart(session) -> None:
    """Configure nntile before collection or any nntile-backed op."""
    del session
    ensure_nntile_context(cpu_fallback=False)


def pytest_sessionfinish(session, exitstatus) -> None:
    """Tear down NNHaul before interpreter finalization (avoids exit UAF)."""
    del session, exitstatus
    if torch_nntile.is_context_initialized():
        torch_nntile.wait()
        torch_nntile.shutdown_context()


def nntile_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Copy an nntile tensor to CPU, flushing a pending TensorGraph first.

    Non-contiguous views are densified first (``contiguous`` records a
    ``Copy``). ``.cpu()`` gathers through NNHaul and blocks until tile data
    is ready; do not add an extra ``wait()`` here. If a pending graph
    exists, compile and run it so the gather sees the produced logical.
    """
    if tensor.device.type == "nntile" and not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if tensor.device.type == "nntile" and torch_nntile.has_pending_graph():
        torch_nntile.compile_graph()
        torch_nntile.run()
    with torch.no_grad():
        return tensor.cpu()
