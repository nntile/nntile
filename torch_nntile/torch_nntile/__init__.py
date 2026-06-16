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
) -> None:
    """Configure StarPU workers before the first libnntile-backed op."""
    _C.init_context(
        ncpu,
        ncuda,
        ooc_enabled,
        ooc_path,
        ooc_size,
        logger,
        verbose,
    )


def is_context_initialized() -> bool:
    return _C.is_context_initialized()


def restrict_cpu() -> None:
    """Pin StarPU codelets to CPU workers (libnntile)."""
    _C.restrict_cpu()


def restrict_cuda() -> None:
    """Pin StarPU codelets to CUDA workers (libnntile)."""
    _C.restrict_cuda()


def restore_where() -> None:
    """Restore default StarPU codelet worker placement."""
    _C.restore_where()


__all__ = [
    "device",
    "_C",
    "init_context",
    "is_context_initialized",
    "restrict_cpu",
    "restrict_cuda",
    "restore_where",
]
