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

__all__ = ["device", "_C"]
