# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/mlp_mixer.py
# MLP-Mixer for device="nntile" - gemm side-L / side-R, no axis swaps.

"""MLP-Mixer matching ``torch_nntile::models::MlpMixer`` / main torch_models.

Input layout is ``[n_patches, batch, patch_dim]`` (``channel_dim = n_patches``).

* Side L (channel mix): ``gemm(x, W, ndim=1, trans_b=True)`` on the last axis.
* Side R (token mix): ``gemm(W, x, ndim=1)`` on the leading axis - no transpose.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from torch_nntile.gemm import gemm
from torch_nntile.nn import LayerNorm
from torch_nntile.nn.functional import add, gelu
from torch_nntile.sum_slice import gap


@dataclass
class MlpMixerConfig:
    channel_dim: int = 8
    init_patch_dim: int = 4
    projected_patch_dim: int = 4
    num_mixer_layers: int = 2
    n_classes: int = 3
    layer_norm_epsilon: float = 1e-5


def _linear_last(x: Tensor, weight: Tensor) -> Tensor:
    """``x @ W.T`` via gemm (W is ``[out, in]``)."""
    return gemm(x, weight, ndim=1, batch_ndim=0, trans_a=False, trans_b=True)


def _linear_leading(weight: Tensor, x: Tensor) -> Tensor:
    """``W @ x`` along axis 0 (W is ``[out, in]``, x is ``[in, ...]``)."""
    return gemm(
        weight, x, ndim=1, batch_ndim=0, trans_a=False, trans_b=False
    )


class MixerMlp(nn.Module):
    """Bias-free expand-4 GELU MLP; side ``L`` last-dim, ``R`` leading-dim."""

    def __init__(self, side: str, dim: int) -> None:
        super().__init__()
        if side not in ("L", "R"):
            raise ValueError("side must be either 'L' or 'R'")
        if dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.side = side
        self.dim = dim
        self.fc1_weight = nn.Parameter(torch.empty(4 * dim, dim))
        self.fc2_weight = nn.Parameter(torch.empty(dim, 4 * dim))
        nn.init.kaiming_uniform_(self.fc1_weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.fc2_weight, a=5**0.5)

    def forward(self, x: Tensor) -> Tensor:
        if self.side == "R":
            h = _linear_leading(self.fc1_weight, x)
            h = gelu(h, approximate="none")
            return _linear_leading(self.fc2_weight, h)
        h = _linear_last(x, self.fc1_weight)
        h = gelu(h, approximate="none")
        return _linear_last(h, self.fc2_weight)


class MixerBlock(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        patch_dim: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm_1 = LayerNorm(patch_dim, eps=eps)
        self.mlp_1 = MixerMlp("R", channel_dim)
        self.norm_2 = LayerNorm(patch_dim, eps=eps)
        self.mlp_2 = MixerMlp("L", patch_dim)

    def forward(self, x: Tensor) -> Tensor:
        y = add(self.mlp_1(self.norm_1(x)), x)
        return add(self.mlp_2(self.norm_2(y)), y)


class MlpMixer(nn.Module):
    """MLP-Mixer classifier over patched tokens (nntile gemm path)."""

    def __init__(self, config: MlpMixerConfig) -> None:
        super().__init__()
        if config.num_mixer_layers < 1:
            raise ValueError("num_mixer_layers must be >= 1")
        self.config = config
        self.stem_weight = nn.Parameter(
            torch.empty(config.projected_patch_dim, config.init_patch_dim)
        )
        self.blocks = nn.ModuleList(
            [
                MixerBlock(
                    config.channel_dim,
                    config.projected_patch_dim,
                    config.layer_norm_epsilon,
                )
                for _ in range(config.num_mixer_layers)
            ]
        )
        self.classifier_weight = nn.Parameter(
            torch.empty(config.n_classes, config.projected_patch_dim)
        )
        nn.init.kaiming_uniform_(self.stem_weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.classifier_weight, a=5**0.5)

    def forward(self, x: Tensor) -> Tensor:
        # x: [n_patches, batch, init_patch_dim]
        h = _linear_last(x, self.stem_weight)
        for block in self.blocks:
            h = block(h)
        # Old ``GAP``: sum_slice(1/P, *, axis=0). Keep [batch, channels]
        # (skip the side-R transpose used by the deleted Linear API).
        pooled = gap(h)
        return _linear_last(pooled, self.classifier_weight)


class MlpMixerCpu(nn.Module):
    """Naive CPU reference matching ``nntile.torch_models.mlp_mixer``.

    Side-R uses ``transpose(0, 2)`` + ``nn.Linear`` (the classical reference);
    the nntile model replaces that with ``gemm(W, x)``.
    """

    def __init__(self, config: MlpMixerConfig) -> None:
        super().__init__()
        self.config = config
        mixer_blocks = [
            _CpuMixer(
                config.channel_dim,
                config.projected_patch_dim,
                config.layer_norm_epsilon,
            )
            for _ in range(config.num_mixer_layers)
        ]
        self.mixer_sequence = nn.Sequential(
            nn.Linear(
                config.init_patch_dim,
                config.projected_patch_dim,
                bias=False,
            ),
            *mixer_blocks,
        )
        self.classification = nn.Linear(
            config.projected_patch_dim, config.n_classes, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.mixer_sequence(x)
        return self.classification(out.mean(dim=0))


class _CpuMixerMlp(nn.Module):
    def __init__(self, side: str, dim: int) -> None:
        super().__init__()
        self.side = side
        self.fn = nn.Sequential(
            nn.Linear(dim, 4 * dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * dim, dim, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.side == "L":
            return self.fn(x)
        x = torch.transpose(x, 0, 2)
        return torch.transpose(self.fn(x), 0, 2)


class _CpuMixer(nn.Module):
    def __init__(
        self, channel_dim: int, patch_dim: int, eps: float
    ) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(patch_dim, eps=eps)
        self.mlp_1 = _CpuMixerMlp("R", channel_dim)
        self.norm_2 = nn.LayerNorm(patch_dim, eps=eps)
        self.mlp_2 = _CpuMixerMlp("L", patch_dim)

    def forward(self, x: Tensor) -> Tensor:
        y = self.mlp_1(self.norm_1(x)) + x
        return self.mlp_2(self.norm_2(y)) + y


def copy_cpu_weights_to_nntile(cpu: MlpMixerCpu, nnt: MlpMixer) -> None:
    """Copy naive-CPU Mixer weights into the nntile gemm model."""
    with torch.no_grad():
        nnt.stem_weight.copy_(cpu.mixer_sequence[0].weight)
        nnt.classifier_weight.copy_(cpu.classification.weight)
        for i, block in enumerate(nnt.blocks):
            src = cpu.mixer_sequence[i + 1]
            block.norm_1.weight.copy_(src.norm_1.weight)
            block.norm_1.bias.copy_(src.norm_1.bias)
            block.norm_2.weight.copy_(src.norm_2.weight)
            block.norm_2.bias.copy_(src.norm_2.bias)
            block.mlp_1.fc1_weight.copy_(src.mlp_1.fn[0].weight)
            block.mlp_1.fc2_weight.copy_(src.mlp_1.fn[2].weight)
            block.mlp_2.fc1_weight.copy_(src.mlp_2.fn[0].weight)
            block.mlp_2.fc2_weight.copy_(src.mlp_2.fn[2].weight)
