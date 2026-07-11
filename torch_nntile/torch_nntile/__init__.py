# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/__init__.py
# Register the PyTorch nntile device (PrivateUse1).

"""PyTorch nntile device stub (PrivateUse1 open registration)."""

from __future__ import annotations

import atexit

import torch

from ._cuda_deps import ensure_linux_cuda_deps

ensure_linux_cuda_deps()

from . import _C  # noqa: E402, F401 — loads kernels and allocator
from . import loss as _loss  # noqa: E402, F401
from . import compat as _compat  # noqa: E402, F401
from . import nn as nn  # noqa: E402, F401
from . import normalization as _normalization  # noqa: E402, F401
from . import norm as _norm  # noqa: E402, F401

_registered = False
_atexit_shutdown_registered = False


def _register_shutdown_atexit() -> None:
    global _atexit_shutdown_registered
    if _atexit_shutdown_registered:
        return
    atexit.register(_shutdown_on_exit)
    _atexit_shutdown_registered = True


def _shutdown_on_exit() -> None:
    if is_context_initialized():
        wait()
        shutdown_context()


class _NntileBackendModule:
    """Minimal torch.nntile runtime module required by PyTorch 2.12+."""

    @staticmethod
    def is_initialized() -> bool:
        return True

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def _is_in_bad_fork() -> bool:
        return False

    @staticmethod
    def manual_seed_all(seed: int) -> None:
        del seed

    @staticmethod
    def device_count() -> int:
        return 1


def _register_backend() -> None:
    global _registered
    if _registered:
        return
    torch.utils.rename_privateuse1_backend("nntile")
    torch.utils.generate_methods_for_privateuse1_backend()
    torch._register_device_module("nntile", _NntileBackendModule)
    _registered = True


_register_backend()

device = torch.device("nntile")


def init_context(
    ncpu: int = -1,
    ncuda: int = -1,
    ooc_enabled: int = 0,
    ooc_path: str = "/tmp/nntile_ooc",
    ooc_size: int = 16 * 1024 * 1024,
    logger: int = 0,
    verbose: int = 0,
    *,
    cpu_fallback: bool = True,
) -> None:
    """Configure StarPU workers before the first libnntile-backed op.

    Records ops into a shared TensorGraph; call :func:`compile_graph`
    and :func:`run` to compile and execute the pending graph.
    """
    _C.init_context(
        ncpu,
        ncuda,
        ooc_enabled,
        ooc_path,
        ooc_size,
        logger,
        verbose,
        cpu_fallback,
    )
    _register_shutdown_atexit()


def execute() -> None:
    """Compile and submit the pending TensorGraph (does **not** wait).

    Equivalent to :func:`compile_graph` then :func:`run`. Call :func:`wait`
    to synchronize and reclaim. Prefer the split API in training loops.
    """
    _C.execute()


def compile_graph() -> None:
    """Lower and compile the pending TensorGraph into the session Runtime."""
    _C.compile_graph()


def run() -> None:
    """Submit the compiled graph to StarPU (asynchronous; does not wait)."""
    _C.run()


def reset_graph_session() -> None:
    """Discard the compiled graph session and recorder state."""
    _C.reset_graph_session()


def has_pending_graph() -> bool:
    return _C.has_pending_graph()


def is_context_initialized() -> bool:
    return _C.is_context_initialized()


def is_cpu_fallback_enabled() -> bool:
    return _C.is_cpu_fallback_enabled()


def restrict_cpu() -> None:
    """Pin StarPU codelets to CPU workers (libnntile)."""
    _C.restrict_cpu()


def restrict_cuda() -> None:
    """Pin StarPU codelets to CUDA workers (libnntile)."""
    _C.restrict_cuda()


def restore_where() -> None:
    """Restore default StarPU codelet worker placement."""
    _C.restore_where()


def wait() -> None:
    """Block until tasks submitted by :func:`run` finish.

    Also runs post-run reclaim (scatter staging invalidate, pin_hold release,
    ``pending_output_reclaim``) and compacts the incremental session so the
    next :func:`compile_graph` stays O(phase) rather than O(history). Call
    before host readout (``.to("cpu")``) or :func:`shutdown_context`.
    Required for clean CUDA teardown when ``ncuda > 0``.
    """
    _C.wait_for_all()


wait_for_all = wait


def shutdown_context() -> None:
    """Shut down libnntile / StarPU and release the global context.

    Flushes any pending TensorGraph and graph session, waits for workers, then
    calls ``Context::shutdown``. Safe to call multiple times. An ``atexit`` hook
    registered by :func:`init_context` runs the same teardown automatically.
    """
    _C.shutdown_context()


def set_axis_group_name(tensor: torch.Tensor, names: dict[int, str]) -> None:
    """Name TensorGraph axis groups for selected dimensions of a tensor.

  Only the listed dimensions are named; others stay unnamed. Names propagate
  to merged axis groups when ops combine tensors. Call before
  :func:`execute` in graph mode.
  """
    _C.set_axis_group_name(tensor, names)


def set_axis_group_tiling(name: str, tile_sizes: int | list[int] | tuple[int, ...]) -> None:
    """Set tiling for a named axis group before :func:`execute`.

  ``tile_sizes`` may be a uniform tile size (``int``) or explicit per-tile
  sizes (``list``/``tuple``) that sum to the axis extent.
  """
    _C.set_axis_group_tiling(name, tile_sizes)


def format_axis_groups() -> str:
    """Return axis-group summary for the pending TensorGraph.

  Format matches the axis-group section of C++ ``TensorGraph::to_string``.
  """
    return _C.format_axis_groups()


def print_axis_groups() -> None:
    """Print axis-group summary for the pending TensorGraph to stdout."""
    _C.print_axis_groups()


def print_info() -> None:
    """Print cumulative ``compile_graph`` / ``run`` / ``wait`` / host-readout timing.

    Useful for comparing nntile overhead against a torch CPU baseline.
    """
    import sys

    sys.stdout.flush()
    _C.print_info()
    sys.stdout.flush()


__all__ = [
    "device",
    "_C",
    "init_context",
    "execute",
    "compile_graph",
    "run",
    "reset_graph_session",
    "has_pending_graph",
    "is_context_initialized",
    "is_cpu_fallback_enabled",
    "restrict_cpu",
    "restrict_cuda",
    "restore_where",
    "wait",
    "wait_for_all",
    "shutdown_context",
    "set_axis_group_name",
    "set_axis_group_tiling",
    "format_axis_groups",
    "print_axis_groups",
    "print_info",
    "nn",
]
