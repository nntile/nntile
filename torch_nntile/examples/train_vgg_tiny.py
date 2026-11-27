#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_vgg_tiny.py
# Tiny VGG-style CNN smoke on cpu / nntile.

"""Tiny VGG-style CNN (stacked Conv / ReLU / MaxPool / Linear).

Exercises repeated ``convolution_overrideable`` + pooling + classifier::

    python torch_nntile/examples/train_vgg_tiny.py train \\
        --device nntile --seed 0 --config vgg_tiny_config.json \\
        --output-dir /tmp/vgg_nntile --steps 1
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


class TinyVGG(nn.Module):
    """Two-stage VGG-like stack (2× Conv per stage) + AdaptiveAvgPool head."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        in_ch = int(cfg["in_channels"])
        channels = [int(c) for c in cfg["channels"]]
        hidden = int(cfg["fc_hidden"])
        num_classes = int(cfg["num_classes"])

        layers: list[nn.Module] = []
        ch = in_ch
        for out_ch in channels:
            layers.extend(
                [
                    nn.Conv2d(ch, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            ch = out_ch
        self.features = nn.Sequential(*layers)
        self.fc1 = nn.Linear(ch, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def _default_config() -> Path:
    return Path(__file__).resolve().parent / "vgg_tiny_config.json"


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
        name="vgg",
        argv=argv,
        default_config=_default_config(),
        model_cls=TinyVGG,
        build_batch=build_batch,
        loss_fn=classification_ce_loss,
        description=__doc__,
    )


if __name__ == "__main__":
    raise SystemExit(main())
