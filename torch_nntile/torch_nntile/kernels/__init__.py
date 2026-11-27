# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/kernels/__init__.py
"""Backward-compatible alias for :mod:`torch_nntile.nn.functional`.

Prefer ``import torch_nntile.nn.functional as nntile_F`` (or
``from torch_nntile.nn.functional import gemm``).
"""

from __future__ import annotations

from torch_nntile.nn import functional

__all__ = list(functional.__all__)

if functional.__all__:
    globals().update({name: getattr(functional, name) for name in __all__})
else:
    def __getattr__(name: str) -> object:
        return getattr(functional, name)
