# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/deep_relu.py
# PyTorch DeepReLU MLP (bias-free Linear + ReLU chain).

"""DeepReLU: a chain of Linear (no bias) -> ReLU blocks.

Matches the NNTile C++ example in ``nntile/examples/deep_relu_forward.cc``:

    input -> [Linear -> ReLU] x (depth - 1) -> Linear -> output
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepReLU(nn.Module):
    """Bias-free deep ReLU network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("DeepReLU: depth must be >= 1")

        layers: list[nn.Module] = []
        in_features = input_dim
        out_features = output_dim if depth == 1 else hidden_dim
        layers.append(nn.Linear(in_features, out_features, bias=False))

        for i in range(1, depth):
            layers.append(nn.ReLU())
            in_features = hidden_dim
            out_features = output_dim if i == depth - 1 else hidden_dim
            layers.append(nn.Linear(in_features, out_features, bias=False))

        self.net = nn.Sequential(*layers)
        self.depth = depth
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @classmethod
    def tiny(cls) -> DeepReLU:
        """Small config used in parity tests (see deep_relu_forward.cc)."""
        return cls(
            input_dim=128,
            hidden_dim=256,
            output_dim=10,
            depth=5,
        )

    @classmethod
    def mnist(
        cls,
        hidden_dim: int = 256,
        depth: int = 5,
    ) -> DeepReLU:
        """MNIST classifier: flattened 28x28 image -> 10 logits."""
        return cls(
            input_dim=28 * 28,
            hidden_dim=hidden_dim,
            output_dim=10,
            depth=depth,
        )

    def init_kaiming_uniform_(self, seed: int = 42) -> None:
        """Kaiming-uniform-style init matching the C++ example."""
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                fan_in = module.in_features
                bound = (1.0 / fan_in) ** 0.5
                module.weight.data.uniform_(-bound, bound, generator=generator)
