# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/nn/__init__.py
# Neural network modules for device="nntile".

"""``torch_nntile.nn`` — NNTile modules/layout helpers and ``functional`` kernels."""

from __future__ import annotations

from torch_nntile.nn import functional
from torch_nntile.nn.sdpa import SDPA, sdpa_eager, sdpa_kernel
from torch_nntile.nn.weight_layout import (
    convert_attn_weights,
    nntile_to_torch_o_weight,
    nntile_to_torch_qkv_weight,
    torch_to_nntile_o_weight,
    torch_to_nntile_qkv_weight,
)

__all__ = [
    "SDPA",
    "convert_attn_weights",
    "functional",
    "nntile_to_torch_o_weight",
    "nntile_to_torch_qkv_weight",
    "sdpa_eager",
    "sdpa_kernel",
    "torch_to_nntile_o_weight",
    "torch_to_nntile_qkv_weight",
]
