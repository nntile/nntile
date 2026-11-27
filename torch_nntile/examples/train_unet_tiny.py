#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_unet_tiny.py
# Tiny U-Net segmentation smoke on cpu / nntile.

"""Tiny U-Net (encoder / decoder / skip ``cat``) on synthetic images.

Upsampling uses ``ConvTranspose2d`` (registered via
``convolution_overrideable``); ``F.interpolate`` is not on PrivateUse1 yet.
Also exercises max-pool, BN, ReLU, and pixel-wise CE::

    python torch_nntile/examples/train_unet_tiny.py train \\
        --device nntile --seed 0 --config unet_tiny_config.json \\
        --output-dir /tmp/unet_nntile --steps 1

    python ... compare --checkpoint-a A.pt --checkpoint-b B.pt
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn_tiny_train_common import (
    make_segmentation_batch,
    run_tiny_cnn_main,
    segmentation_ce_loss,
)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(x)))


class TinyUNet(nn.Module):
    """Minimal U-Net with ``depth`` down/up stages and skip concatenations."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        in_ch = int(cfg["in_channels"])
        base = int(cfg["base_channels"])
        depth = int(cfg["depth"])
        num_classes = int(cfg["num_classes"])
        if depth < 1:
            raise ValueError("unet depth must be >= 1")

        downs: list[nn.Module] = []
        ch = in_ch
        out_ch = base
        for _ in range(depth):
            downs.append(ConvBlock(ch, out_ch))
            ch = out_ch
            out_ch *= 2
        self.downs = nn.ModuleList(downs)

        bottleneck_ch = base * (2 ** (depth - 1))
        self.bottleneck = ConvBlock(bottleneck_ch, bottleneck_ch * 2)

        ups: list[nn.Module] = []
        up_convs: list[nn.Module] = []
        ch = bottleneck_ch * 2
        for i in range(depth):
            skip_ch = base * (2 ** (depth - 1 - i))
            ups.append(
                nn.ConvTranspose2d(ch, skip_ch, kernel_size=2, stride=2)
            )
            up_convs.append(ConvBlock(skip_ch * 2, skip_ch))
            ch = skip_ch
        self.ups = nn.ModuleList(ups)
        self.up_convs = nn.ModuleList(up_convs)
        self.head = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x):
        skips: list = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, up_conv, skip in zip(
            self.ups,
            self.up_convs,
            reversed(skips),
        ):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = up_conv(x)
        return self.head(x)


def _default_config() -> Path:
    return Path(__file__).resolve().parent / "unet_tiny_config.json"


def main(argv: list[str] | None = None) -> int:
    def build_batch(cfg: dict[str, Any], args):
        return make_segmentation_batch(
            batch_size=args.batch_size,
            channels=int(cfg["in_channels"]),
            height=int(cfg["height"]),
            width=int(cfg["width"]),
            num_classes=int(cfg["num_classes"]),
            seed=args.seed if args.seed is not None else 0,
        )

    return run_tiny_cnn_main(
        name="unet",
        argv=argv,
        default_config=_default_config(),
        model_cls=TinyUNet,
        build_batch=build_batch,
        loss_fn=segmentation_ce_loss,
        description=__doc__,
    )


if __name__ == "__main__":
    raise SystemExit(main())
