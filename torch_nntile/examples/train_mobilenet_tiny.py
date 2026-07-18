#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_mobilenet_tiny.py
# Tiny MobileNet-style CNN smoke on cpu / nntile.

"""Tiny MobileNet-style CNN (depthwise separable conv + BN).

Exercises grouped ``Conv2d`` (``groups=C``) via
``convolution_overrideable``, plus pointwise 1×1 and AdaptiveAvgPool::

    python torch_nntile/examples/train_mobilenet_tiny.py train \\
        --device nntile --seed 0 --config mobilenet_tiny_config.json \\
        --output-dir /tmp/mobilenet_nntile --steps 1
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


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_ch,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.dw(x)))
        return F.relu(self.bn2(self.pw(x)))


class TinyMobileNet(nn.Module):
    """Stem + a few depthwise-separable blocks + global pool classifier."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        in_ch = int(cfg["in_channels"])
        base = int(cfg["base_channels"])
        blocks = int(cfg["blocks"])
        num_classes = int(cfg["num_classes"])
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_ch,
                base,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        layers: list[nn.Module] = []
        ch = base
        for i in range(blocks):
            out_ch = base * (2 if i > 0 else 1)
            # First block keeps spatial size; later blocks downsample.
            stride = 1 if i == 0 else 2
            layers.append(DepthwiseSeparable(ch, out_ch, stride=stride))
            ch = out_ch
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Linear(ch, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


def _default_config() -> Path:
    return Path(__file__).resolve().parent / "mobilenet_tiny_config.json"


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
        name="mobilenet",
        argv=argv,
        default_config=_default_config(),
        model_cls=TinyMobileNet,
        build_batch=build_batch,
        loss_fn=classification_ce_loss,
        description=__doc__,
    )


if __name__ == "__main__":
    raise SystemExit(main())
