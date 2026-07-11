#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/reproduce_google_five_layer_relu_mnist.py
# Reproduce Google "TensorFlow without a PhD" five-layer ReLU MNIST.

"""Reproduce Google's five-layer ReLU MNIST digit-recognition experiment.

Source of truth (TensorFlow)::

    https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/blob/master/tensorflow-mnist-tutorial/mnist_2.1_five_layers_relu_lrdecay.py

Architecture: ``784 → 200 → 100 → 60 → 30 → 10`` with ReLU hidden layers and
raw logits on the output. Loss is mean softmax cross-entropy. Optimizer is
Adam with per-step learning rate::

    lr(step) = 0.0001 + 0.003 * exp(-step / 2000)

Default schedule: batch size 100, 10 000 steps (~16.7 epochs over 60 000
training images). The Google script documents **final test accuracy ≈ 0.9824**
with this recipe.

This is a pure PyTorch CPU (or CUDA) reference reproduction — it does not use
``device="nntile"``. For the existing full-batch nntile DeepReLU parity smoke,
see ``train_deep_relu_mnist.py``.

Example::

    python torch_nntile/examples/reproduce_google_five_layer_relu_mnist.py \\
        --steps 10000

Success floor for this reproduction: test accuracy ≥ 0.97 (prefer ~0.975–0.985).

Observed (CPU, seed=0, 10 000 steps): max test accuracy **0.9827**, final
**0.9822** (Google source quotes ≈ 0.9824).
"""

from __future__ import annotations

import argparse
import math
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


HIDDEN_WIDTHS = (200, 100, 60, 30)


class FiveLayerReLU(nn.Module):
    """Bias-aware MLP matching the Google five-layer ReLU tutorial."""

    def __init__(self) -> None:
        super().__init__()
        widths = (28 * 28, *HIDDEN_WIDTHS, 10)
        layers: list[nn.Linear] = []
        for in_features, out_features in zip(widths[:-1], widths[1:]):
            layers.append(nn.Linear(in_features, out_features))
        self.layers = nn.ModuleList(layers)
        self._init_google_()

    def _init_google_(self) -> None:
        """Match TF truncated_normal(σ=0.1) ≈ N(0, 0.1); ReLU biases 0.1."""
        for index, layer in enumerate(self.layers):
            nn.init.normal_(layer.weight, mean=0.0, std=0.1)
            if index < len(self.layers) - 1:
                nn.init.constant_(layer.bias, 0.1)
            else:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


def learning_rate(step: int) -> float:
    """Google exponential decay: 0.0001 + 0.003 * exp(-step / 2000)."""
    return 0.0001 + 0.003 * math.exp(-step / 2000.0)


def infinite_batches(
    loader: DataLoader,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from loader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        total_loss += float(F.cross_entropy(logits, labels, reduction="sum"))
        total_correct += int((logits.argmax(dim=1) == labels).sum())
        total += labels.numel()
    model.train()
    return total_loss / total, total_correct / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="/tmp/mnist_google_five_layer_relu",
        help="Directory for torchvision MNIST download",
    )
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (default: cpu)",
    )
    parser.add_argument(
        "--train-log-every",
        type=int,
        default=20,
        help="Print train minibatch metrics every N steps",
    )
    parser.add_argument(
        "--test-every",
        type=int,
        default=100,
        help="Evaluate full test set every N steps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    transform = transforms.ToTensor()
    train_set = datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.MNIST(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1000,
        shuffle=False,
    )

    model = FiveLayerReLU().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate(0))
    batches = infinite_batches(train_loader)

    print(
        "Reproducing Google five-layer ReLU MNIST "
        f"(steps={args.steps}, batch={args.batch_size}, device={device})"
    )
    print(
        "Source expected test accuracy ≈ 0.9824; "
        "reproduction success floor ≥ 0.97"
    )

    max_test_acc = 0.0
    last_test_loss = float("nan")
    last_test_acc = float("nan")
    model.train()

    for step in range(args.steps + 1):
        lr = learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        images, labels = next(batches)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        if step % args.train_log_every == 0:
            with torch.no_grad():
                train_acc = float(
                    (logits.argmax(dim=1) == labels).float().mean()
                )
            print(
                f"{step}: train accuracy={train_acc:.4f} "
                f"loss={float(loss.detach()):.4f} (lr={lr:.6f})"
            )

        if step % args.test_every == 0:
            test_loss, test_acc = evaluate(model, test_loader, device)
            last_test_loss = test_loss
            last_test_acc = test_acc
            max_test_acc = max(max_test_acc, test_acc)
            epoch = step * args.batch_size // len(train_set) + 1
            print(
                f"{step}: ********* epoch {epoch} ********* "
                f"test accuracy={test_acc:.4f} test loss={test_loss:.4f}"
            )

    print(f"max test accuracy: {max_test_acc:.4f}")
    print(
        f"final test accuracy: {last_test_acc:.4f} "
        f"final test loss: {last_test_loss:.4f}"
    )
    if last_test_acc < 0.97:
        raise SystemExit(
            f"Reproduction failed: final test accuracy {last_test_acc:.4f} "
            f"< 0.97 (Google source quotes ≈ 0.9824)"
        )


if __name__ == "__main__":
    main()
