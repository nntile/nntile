#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_unet_modern_tiny.py
# Modern U-Net (F.interpolate) smoke on cpu / nntile.

"""Modern tiny U-Net: upsample via ``F.interpolate`` (not ConvTranspose).

Uses bilinear (default) or nearest ``mode`` from JSON ``upsample_mode``.
Skip connections still ``torch.cat``. Exercises newly registered
``upsample_bilinear2d`` / ``upsample_nearest2d`` (+ backward)::

    python torch_nntile/examples/train_unet_modern_tiny.py train \\
        --device nntile --seed 0 --config unet_modern_tiny_config.json \\
        --output-dir /tmp/unet_modern_nntile --steps 1
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


class TinyModernUNet(nn.Module):
    """U-Net with ``F.interpolate`` up path (no learnable transpose)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        in_ch = int(cfg["in_channels"])
        base = int(cfg["base_channels"])
        depth = int(cfg["depth"])
        num_classes = int(cfg["num_classes"])
        mode = str(cfg.get("upsample_mode", "bilinear"))
        if mode not in ("bilinear", "nearest"):
            raise ValueError(
                f"upsample_mode must be bilinear|nearest, got {mode!r}"
            )
        self.upsample_mode = mode
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

        up_convs: list[nn.Module] = []
        # After interpolate, channels still match bottleneck / previous;
        # 1x1 reduces before cat+ConvBlock for a lighter modern style.
        reduce: list[nn.Module] = []
        ch = bottleneck_ch * 2
        for i in range(depth):
            skip_ch = base * (2 ** (depth - 1 - i))
            reduce.append(nn.Conv2d(ch, skip_ch, kernel_size=1))
            up_convs.append(ConvBlock(skip_ch * 2, skip_ch))
            ch = skip_ch
        self.reduce = nn.ModuleList(reduce)
        self.up_convs = nn.ModuleList(up_convs)
        self.head = nn.Conv2d(base, num_classes, kernel_size=1)

    def _upsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample_mode == "nearest":
            return F.interpolate(x, scale_factor=2, mode="nearest")
        return F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x):
        skips: list = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for reduce, up_conv, skip in zip(
            self.reduce,
            self.up_convs,
            reversed(skips),
        ):
            x = self._upsample(x)
            x = reduce(x)
            x = torch.cat([x, skip], dim=1)
            x = up_conv(x)
        return self.head(x)


def _default_config() -> Path:
    return Path(__file__).resolve().parent / "unet_modern_tiny_config.json"


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
        name="unet_modern",
        argv=argv,
        default_config=_default_config(),
        model_cls=TinyModernUNet,
        build_batch=build_batch,
        loss_fn=segmentation_ce_loss,
        description=__doc__,
    )


if __name__ == "__main__":
    raise SystemExit(main())
