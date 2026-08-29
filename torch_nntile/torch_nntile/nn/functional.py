# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/nn/functional.py
"""NNTile-specific functional kernels (not stock ``torch.nn.functional``).

Stock ``torch.nn`` / ``torch.nn.functional`` on ``device=nntile`` follow the
CUDA dispatch path (torch-native ``aten`` codelets). Use **this** module for
classic NNTile tiled / hand-written ops (``gemm``, ``rope``, ``sum_slice``,
…).
"""

from __future__ import annotations

from torch_nntile.add_fiber import add_fiber
from torch_nntile.gemm import gemm, matmul
from torch_nntile.loss import cross_entropy, mse_loss
from torch_nntile.norm import vector_norm
from torch_nntile.nn.activations import gelu, relu, silu
from torch_nntile.nn.add import add
from torch_nntile.nn.embedding import embedding
from torch_nntile.nn.mul import mul, mul_scalar
from torch_nntile.nn.norm import layer_norm, rms_norm
from torch_nntile.nn.sdpa import sdpa_eager, sdpa_kernel
from torch_nntile.rope import rope, rope_sin_cos_from_position_ids
from torch_nntile.sum_slice import gap, sum_slice

__all__ = [
    "add",
    "add_fiber",
    "apply_nntile_patches",
    "cross_entropy",
    "embedding",
    "gap",
    "gelu",
    "gemm",
    "layer_norm",
    "matmul",
    "mul",
    "mul_scalar",
    "mse_loss",
    "relu",
    "rms_norm",
    "rope",
    "rope_sin_cos_from_position_ids",
    "sdpa_eager",
    "sdpa_kernel",
    "silu",
    "sum_slice",
    "vector_norm",
]


def apply_nntile_patches() -> None:
    """Opt-in legacy shim: override selected stock ``F.*`` / ``linalg`` symbols.

    New code should call :func:`rms_norm`, :func:`cross_entropy`, etc. from
    this module explicitly, or use stock ``torch.nn.functional`` on
    ``device=nntile`` (CUDA parity). Do **not** call this in torch-native
    training stacks.
    """
    from torch_nntile.loss import patch_cross_entropy
    from torch_nntile.norm import patch_vector_norm
    from torch_nntile.normalization import patch_rms_norm

    patch_cross_entropy()
    patch_vector_norm()
    patch_rms_norm()
