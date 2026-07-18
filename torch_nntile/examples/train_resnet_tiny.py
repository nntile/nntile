#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_resnet_tiny.py
# Tiny ResNet-style CNN smoke on cpu / nntile.

"""Tiny ResNet-style CNN (Conv / BN / ReLU / residual / AdaptiveAvgPool).

Exercises StarPU-backed ``convolution_overrideable``,
``native_batch_norm``, ``_adaptive_avg_pool2d``, and residual ``add``::

    python torch_nntile/examples/train_resnet_tiny.py train \\
        --device nntile --seed 0 --config resnet_tiny_config.json \\
        --output-dir /tmp/resnet_nntile --steps 1

    python ... compare --checkpoint-a A.pt --checkpoint-b B.pt
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn as nn
import torch.nn.functional as F

from cnn_tiny_train_common import (
    classification_ce_loss,
    make_image_batch,
    run_tiny_cnn_main,
)


class BasicBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class TinyResNet(nn.Module):
    """Minimal residual CNN (no downsample stem complexity)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        in_ch = int(cfg["in_channels"])
        base = int(cfg["base_channels"])
        blocks = int(cfg["blocks"])
        num_classes = int(cfg["num_classes"])
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.Sequential(
            *[BasicBlock(base) for _ in range(blocks)]
        )
        self.head = nn.Linear(base, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


def _default_config() -> Path:
    return Path(__file__).resolve().parent / "resnet_tiny_config.json"


def main(argv: list[str] | None = None) -> int:
    def build_batch(cfg: dict[str, Any], args):
        return make_image_batch(
            batch_size=args.batch_size,
            channels=int(cfg["in_channels"]),
            height=int(cfg["height"]),
            width=int(cfg["width"]),
            num_classes=int(cfg["num_classes"]),
            seed=args.seed if args.seed is not None else 0,
        )

    return run_tiny_cnn_main(
        name="resnet",
        argv=argv,
        default_config=_default_config(),
        model_cls=TinyResNet,
        build_batch=build_batch,
        loss_fn=classification_ce_loss,
        description=__doc__,
    )


if __name__ == "__main__":
    raise SystemExit(main())
