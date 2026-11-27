# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/nn/__init__.py
# Neural network modules for device="nntile".

"""``torch_nntile.nn`` — classic NNTile kernels on ``device=nntile``.

Stock ``torch.nn`` / ``torch.nn.functional`` on ``device=nntile`` use
torch-native ATen StarPU codelets (PyTorch autograd). This package is the
hand-written nntile kernel path:

* :mod:`torch_nntile.nn.functional` — autograd functions / functional API
* :mod:`torch_nntile.nn.module` — ``torch.nn.Module`` subclasses
* :mod:`torch_nntile.nn.model` — models built from those modules
"""

from __future__ import annotations

from . import functional, module
from .module import (
    CrossEntropyLoss,
    Embedding,
    GELU,
    LayerNorm,
    Linear,
    NntileAttentionOutput,
    NntileLinear,
    NntileQKVProjection,
    RMSNorm,
    ReLU,
    SDPA,
    SiLU,
)
from .sdpa import sdpa_eager, sdpa_kernel
from .weight_layout import (
    convert_attn_weights,
    nntile_to_torch_o_weight,
    nntile_to_torch_qkv_weight,
    torch_to_nntile_o_weight,
    torch_to_nntile_qkv_weight,
)

__all__ = [
    "CrossEntropyLoss",
    "Embedding",
    "GELU",
    "LayerNorm",
    "Linear",
    "NntileAttentionOutput",
    "NntileLinear",
    "NntileQKVProjection",
    "RMSNorm",
    "ReLU",
    "SDPA",
    "SiLU",
    "convert_attn_weights",
    "functional",
    "module",
    "nntile_to_torch_o_weight",
    "nntile_to_torch_qkv_weight",
    "sdpa_eager",
    "sdpa_kernel",
    "torch_to_nntile_o_weight",
    "torch_to_nntile_qkv_weight",
]
