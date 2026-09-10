# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/nn/modules.py
"""``torch.nn``-style modules backed by classic NNTile kernels."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from torch_nntile.loss import cross_entropy as _cross_entropy
from torch_nntile.nn.activations import gelu, relu, silu
from torch_nntile.nn.embedding import embedding as _embedding
from torch_nntile.nn.linear import NntileLinear
from torch_nntile.nn.norm import layer_norm, rms_norm


class Linear(NntileLinear):
    """Alias for :class:`NntileLinear`."""


class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


class GELU(nn.Module):
    def __init__(self, approximate: str = "tanh") -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        return gelu(x, approximate=self.approximate)


class SiLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return silu(x)


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        else:
            normalized_shape = tuple(normalized_shape)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(normalized_shape))
            self.bias = nn.Parameter(torch.empty(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        return layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )


class RMSNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        eps: float | None = None,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        else:
            normalized_shape = tuple(normalized_shape)
        self.normalized_shape = normalized_shape
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(normalized_shape))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.ones_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.normalized_shape, self.weight, self.eps)


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Classic kernel ignores padding; expose the torch.nn attribute so
        # HF-parity checks and loaders can read it (always unused / None).
        self.padding_idx: int | None = None
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)

    def forward(self, indices: Tensor) -> Tensor:
        return _embedding(self.weight, indices)


class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return _cross_entropy(
            logits,
            target,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )


__all__ = [
    "CrossEntropyLoss",
    "Embedding",
    "GELU",
    "LayerNorm",
    "Linear",
    "RMSNorm",
    "ReLU",
    "SiLU",
]
