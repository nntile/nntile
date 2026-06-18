# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/__init__.py
# Register the PyTorch nntile device (PrivateUse1).

"""PyTorch nntile device stub (PrivateUse1 open registration)."""

from __future__ import annotations

import torch

from . import _C  # noqa: F401 — loads kernels and allocator

_registered = False


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
    runtime_mode: str = "eager",
) -> None:
    """Configure StarPU workers before the first libnntile-backed op.

    ``runtime_mode`` is ``"eager"`` (compile and run each op immediately) or
    ``"graph"`` (record ops into a shared TensorGraph; call :func:`execute`
    to compile and run the pending graph).
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
        runtime_mode,
    )


def execute() -> None:
    """Compile and run the pending TensorGraph, then reset the recorder.

    Required in graph mode before reading nntile tensor data on the host.
    No-op when the pending graph is empty (including in eager mode).
    """
    _C.execute()


def is_graph_mode() -> bool:
    return _C.is_graph_mode()


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


__all__ = [
    "device",
    "_C",
    "init_context",
    "execute",
    "is_graph_mode",
    "has_pending_graph",
    "is_context_initialized",
    "is_cpu_fallback_enabled",
    "restrict_cpu",
    "restrict_cuda",
    "restore_where",
    "set_axis_group_name",
    "set_axis_group_tiling",
    "format_axis_groups",
    "print_axis_groups",
]
