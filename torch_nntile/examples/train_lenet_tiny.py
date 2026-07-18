#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_lenet_tiny.py
# Tiny LeNet-style CNN smoke on cpu / nntile.

"""Tiny LeNet-style CNN (Conv / ReLU / MaxPool / Linear) on synthetic images.

Exercises StarPU-backed ``convolution_overrideable``,
``max_pool2d_with_indices``, and linear/matmul primitives::

    python torch_nntile/examples/train_lenet_tiny.py train \\
        --device nntile --seed 0 --config lenet_tiny_config.json \\
        --output-dir /tmp/lenet_nntile --steps 1

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


class TinyLeNet(nn.Module):
    """Minimal LeNet-like stack (no dropout)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        c1 = int(cfg["conv1_channels"])
        c2 = int(cfg["conv2_channels"])
        hidden = int(cfg["fc_hidden"])
        num_classes = int(cfg["num_classes"])
        in_ch = int(cfg["in_channels"])
        self.conv1 = nn.Conv2d(in_ch, c1, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(c2 * 7 * 7, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def _default_config() -> Path:
    return Path(__file__).resolve().parent / "lenet_tiny_config.json"


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
        name="lenet",
        argv=argv,
        default_config=_default_config(),
        model_cls=TinyLeNet,
        build_batch=build_batch,
        loss_fn=classification_ce_loss,
        description=__doc__,
    )


if __name__ == "__main__":
    raise SystemExit(main())
