#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_deep_relu_mnist.py
# Train DeepReLU on the full MNIST training set (60k batch) on CPU vs nntile.

"""Full-batch MNIST training with DeepReLU on CPU and device=\"nntile\".

Cross-entropy is evaluated on nntile via ``torch_nntile.training.cross_entropy``
(same tensor-op chain as ``NNCrossEntropyOp`` in libnntile). The scalar loss lives
on ``device="nntile"``; call ``torch_nntile.execute()`` in graph mode before
reading it on the host.

Example::

    export LD_LIBRARY_PATH=$PWD/build/nntile:/opt/starpu/lib
    python torch_nntile/examples/train_deep_relu_mnist.py --epochs 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision import datasets

# Allow running from repo root before editable install.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "torch_nntile") not in sys.path:
    sys.path.insert(0, str(_REPO / "torch_nntile"))

import torch_nntile  # noqa: E402
from torch_nntile import _C  # noqa: E402
from torch_nntile.models import DeepReLU  # noqa: E402
from torch_nntile.training import (  # noqa: E402
    clone_model_weights,
    max_weight_delta,
    train_full_batch_step,
)


def load_mnist_full_batch(
    data_dir: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flattened training images ``[60000, 784]`` and labels ``[60000]``."""
    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
    )
    images = dataset.data.reshape(len(dataset), -1).to(torch.float32) / 255.0
    labels = dataset.targets.clone()
    return images, labels


def build_models(
    seed: int,
    hidden_dim: int,
    depth: int,
) -> tuple[DeepReLU, DeepReLU]:
    torch.manual_seed(seed)
    model_cpu = DeepReLU.mnist(hidden_dim=hidden_dim, depth=depth)
    model_cpu.init_kaiming_uniform_(seed=seed)

    model_nnt = DeepReLU.mnist(hidden_dim=hidden_dim, depth=depth)
    model_nnt.load_state_dict(model_cpu.state_dict())
    model_nnt = model_nnt.to("nntile")
    return model_cpu, model_nnt


def train_on_device(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: str,
    epochs: int,
    learning_rate: float,
) -> list[float]:
    if device == "nntile":
        x = images.to("nntile")
    else:
        x = images
    y = labels if device == "cpu" else labels  # CE bridge expects CPU targets

    losses: list[float] = []
    for epoch in range(epochs):
        loss = train_full_batch_step(model, x, y, learning_rate)
        losses.append(loss)
        print(f"[{device}] epoch {epoch + 1}/{epochs}  loss={loss:.6f}")
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/tmp/mnist_torch_nntile")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument(
        "--runtime-mode",
        choices=("eager", "graph"),
        default="eager",
        help="torch_nntile runtime mode (default: eager)",
    )
    parser.add_argument("--output-dir", default="deep_relu_mnist_runs")
    args = parser.parse_args()

    if not _C.has_libnntile():
        raise SystemExit(
            "torch_nntile was built without libnntile. "
            "Set NNTILE_BUILD_DIR and reinstall."
        )

    torch_nntile.init_context(
        ncpu=1,
        ncuda=0,
        verbose=0,
        cpu_fallback=False,
        runtime_mode=args.runtime_mode,
    )

    print(f"Runtime mode: {args.runtime_mode}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MNIST training set (60 000 images, single batch)...")
    images, labels = load_mnist_full_batch(args.data_dir)
    print(f"  images {tuple(images.shape)}, labels {tuple(labels.shape)}")

    model_cpu, model_nnt = build_models(
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    )

    init_cpu = clone_model_weights(model_cpu)
    init_nnt = clone_model_weights(model_nnt)
    init_delta = max_weight_delta(init_cpu, init_nnt)
    print(f"Initial weight max |cpu - nntile| = {init_delta:.3e} (expect 0)")

    print("\nTraining on CPU...")
    cpu_losses = train_on_device(
        model_cpu,
        images,
        labels,
        device="cpu",
        epochs=args.epochs,
        learning_rate=args.lr,
    )

    print("\nTraining on nntile...")
    nnt_losses = train_on_device(
        model_nnt,
        images,
        labels,
        device="nntile",
        epochs=args.epochs,
        learning_rate=args.lr,
    )

    cpu_path = output_dir / "deep_relu_mnist_cpu.pt"
    nnt_path = output_dir / "deep_relu_mnist_nntile.pt"
    torch.save(model_cpu.state_dict(), cpu_path)
    torch.save(clone_model_weights(model_nnt), nnt_path)

    final_cpu = clone_model_weights(model_cpu)
    final_nnt = clone_model_weights(model_nnt)
    weight_delta = max_weight_delta(final_cpu, final_nnt)

    print("\nLoss comparison (cpu vs nntile):")
    for epoch, (loss_cpu, loss_nnt) in enumerate(
        zip(cpu_losses, nnt_losses), start=1
    ):
        print(
            f"  epoch {epoch}: cpu={loss_cpu:.6f}  nntile={loss_nnt:.6f}  "
            f"diff={abs(loss_cpu - loss_nnt):.3e}"
        )

    print(f"\nFinal weight max |cpu - nntile| = {weight_delta:.3e}")
    print(f"Saved CPU model to {cpu_path}")
    print(f"Saved nntile model (CPU tensors) to {nnt_path}")


if __name__ == "__main__":
    main()
