# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/__init__.py
# Register the PyTorch nntile device (PrivateUse1).

"""PyTorch nntile device (PrivateUse1; requires libnntile).

Stock ``torch.nn`` on ``device=nntile`` uses torch-native kernels.
Classic nntile kernels live under :mod:`torch_nntile.nn` (``functional``,
``module``, ``model``). Autograd is PyTorch's.
"""

from __future__ import annotations

import atexit
import os

import torch

from ._build_info import BUILT_WITH_CUDA, NNTILE_NATIVE_OPS, TORCH_NATIVE_OPS
from ._cuda_deps import ensure_linux_cuda_deps

ensure_linux_cuda_deps(required=BUILT_WITH_CUDA)

# Stock torch.nn on device=nntile uses torch-native TORCH_* codelets when
# TORCH_NATIVE_OPS is on. Classic nntile::kernel ops live under
# torch_nntile.nn (functional / module / model) when NNTILE_NATIVE_OPS is on.
from . import _C  # noqa: E402, F401 - loads kernels and allocator
from . import (  # noqa: E402, F401; noqa: E402, backward-compat alias
    compat as _compat, kernels as kernels, nn as nn)

_registered = False
_atexit_shutdown_registered = False


def built_with_cuda() -> bool:
    """Return whether this install was compiled with CUDA support."""
    return bool(BUILT_WITH_CUDA)


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

    @staticmethod
    def get_amp_supported_dtype() -> list[torch.dtype]:
        # RoPE / HF disable autocast around float32 math; torch still
        # requires this hook when device_type is ``nntile``.
        return [torch.float16, torch.bfloat16]


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

# Opt-in local TensorGraph → TileGraph compiler (Dana). Platform kernels
# leave this unset so compile_graph(), execute(), and .to("cpu") auto-flush
# fail by default.
_LOCAL_COMPILER_ENV = "NNTILE_ENABLE_LOCAL_COMPILER"
_LOCAL_COMPILER_DISABLED_MSG = (
    "TensorGraph compilation is disabled by default. "
    "Set NNTILE_ENABLE_LOCAL_COMPILER=1 or use the NNTile "
    "platform."
)


def local_compiler_enabled() -> bool:
    """Return whether this process may lower a TensorGraph locally.

    Requires ``NNTILE_ENABLE_LOCAL_COMPILER=1``. Any other value, including
    unset, keeps the local compiler off. Gates public :func:`compile_graph`,
    legacy :func:`execute`, and C++ ``.to("cpu")`` auto-flush.
    """
    return os.environ.get(_LOCAL_COMPILER_ENV, "") == "1"


def _require_local_compiler() -> None:
    if _C.platform_session_active():
        raise RuntimeError(
            "the server compiles; use .to('cpu') or torch_nntile.wait() "
            "(local compile is disabled on the platform)"
        )
    if not local_compiler_enabled():
        raise RuntimeError(_LOCAL_COMPILER_DISABLED_MSG)


def init_context(
    ncpu: int = -1,
    ncuda: int = -1,
    ooc_enabled: int = 0,
    ooc_path: str = "/tmp/nntile_ooc",
    ooc_size: int = 16 * 1024 * 1024,
    logger: int = 0,
    verbose: int = 0,
    *,
    cpu_fallback: bool = False,
    account_id: str | None = None,
) -> None:
    """Configure StarPU workers, or connect to nntile-server.

    When ``NNTILE_SERVER_SOCKET`` is set, this process is a platform
    kernel: no local StarPU, no hardware knobs. ``.to("nntile")``
    Ingresses bytes and ``.to("cpu")`` Flushes the recorded TensorGraph.

    Otherwise records ops into a shared TensorGraph; call
    :func:`compile_graph` or :func:`execute` (both require
    ``NNTILE_ENABLE_LOCAL_COMPILER=1``) to compile on this process.

    ``cpu_fallback`` defaults to False: unregistered aten ops raise
    instead of silently copying nntile tensors to CPU. Move data only
    with ``.to("nntile")`` / ``.to("cpu")``.
    """
    socket = os.environ.get("NNTILE_SERVER_SOCKET", "").strip()
    if socket:
        from . import _platform

        _platform.connect(account_id=account_id)
        _register_shutdown_atexit()
        return
    del account_id
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

    Off by default: same ``NNTILE_ENABLE_LOCAL_COMPILER=1`` gate as
    :func:`compile_graph`.
    """
    _require_local_compiler()
    _C.execute()


def compile_graph() -> None:
    """Lower and compile the pending TensorGraph into the session Runtime.

    Off by default. Set ``NNTILE_ENABLE_LOCAL_COMPILER=1`` to compile
    locally (researcher on their own box). Without that exact value this
    raises ``RuntimeError``; use the NNTile platform to compile on a
    shared node. Legacy :func:`execute` and host ``.to("cpu")`` auto-flush
    use the same gate so a shared-node kernel cannot lower locally.

    Does **not** wait for a prior :func:`run`. The next phase may be sealed
    and submitted while StarPU is still executing an earlier phase; call
    :func:`wait` when host-side results or reclaim are required.
    """
    _require_local_compiler()
    _C.compile_graph()


def run() -> None:
    """Submit the compiled graph to StarPU (asynchronous; does not wait).

    Reclaim is ordinary ``INVALIDATE`` ops in the submitted stream: last
    ``TensorRef`` drop records ``tensor::invalidate``, and
    :func:`compile_graph` also appends INVALIDATE for unmarked phase temps.
    Free the step autograd graph (``del loss`` after ``loss.detach()``) before
    :func:`compile_graph` so those temps are unmarked. ``del`` of inputs after
    their last recorded use is safe (invalidate is ordered after that use).
    Only :func:`wait` joins StarPU (host readout / shutdown).
    """
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
    """Join StarPU after :func:`run` for host-visible completion.

    Pin release / async tile reclaim already ran at the end of :func:`run`.
    Compacts TensorGraph history so the next :func:`compile_graph` stays
    O(phase). Call before host readout (``.to("cpu")``) or
    :func:`shutdown_context`. Required for clean CUDA teardown when
    ``ncuda > 0``.
    """
    _C.wait_for_all()


wait_for_all = wait


def shutdown_context() -> None:
    """Shut down libnntile / StarPU and release the global context.

    Flushes any pending TensorGraph and graph session, waits for workers, then
    calls ``Context::shutdown``. Safe to call multiple times. An ``atexit`` hook
    registered by :func:`init_context` runs the same teardown automatically.
    """
    if _C.platform_session_active():
        from . import _platform

        _platform.teardown()
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

    Temporarily raises if a torch-native (``TORCH_*``) compute op is in
    the pending graph: stock aten on ``device=nntile`` stays untiled.
    Classic ``torch_nntile.nn`` graphs may tile. See
    ``docs/dev/torch_nntile_aten_ops.md``.
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


def pending_op_names() -> list[str]:
    """Return op names in the pending TensorGraph phase (classic vs TORCH_*)."""
    return list(_C.pending_op_names())


def format_pending_data_sizes() -> str:
    """Pending TensorGraph data nbytes grouped by tensor name."""
    return str(_C.format_pending_data_sizes())


def print_info() -> None:
    """Print cumulative ``compile_graph`` / ``run`` / ``wait`` / host-readout timing.

    Useful for comparing nntile overhead against a torch CPU baseline.
    """
    import sys

    sys.stdout.flush()
    _C.print_info()
    sys.stdout.flush()


def record_nntile_seconds() -> float:
    """Cumulative nntile record time in seconds (``record(nntile)``).

    Snapshot around a train-step record window. Remaining record wall is
    PyTorch overhead (``record(torch)``): Python, autograd, and dispatch
    into nntile kernels.
    """
    return float(_C.record_nntile_seconds())


__all__ = [
    "device",
    "_C",
    "built_with_cuda",
    "BUILT_WITH_CUDA",
    "TORCH_NATIVE_OPS",
    "NNTILE_NATIVE_OPS",
    "init_context",
    "execute",
    "local_compiler_enabled",
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
    "pending_op_names",
    "format_pending_data_sizes",
    "print_info",
    "record_nntile_seconds",
    "nn",
    "kernels",
]
