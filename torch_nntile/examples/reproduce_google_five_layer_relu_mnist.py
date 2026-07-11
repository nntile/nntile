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

Supports ``--device cpu`` / ``cuda`` (PyTorch Adam) and ``--device nntile``
(``torch_nntile.training.Adam`` + ``cross_entropy``, graph compile/run per
step). ``nn.Linear`` bias is unsupported on nntile, so layers use
``F.linear(x, weight, None) + bias``.

CPU torch reference::

    python torch_nntile/examples/reproduce_google_five_layer_relu_mnist.py \\
        --device cpu --steps 10000

Nntile (CPU StarPU workers)::

    export LD_LIBRARY_PATH=$PWD/build/nntile:/opt/starpu/lib
    STARPU_NCPU=4 STARPU_NCUDA=0 \\
    python torch_nntile/examples/reproduce_google_five_layer_relu_mnist.py \\
        --device nntile --steps 10000

Success floor: test accuracy ≥ 0.97 (prefer ~0.975–0.985).

Observed (``--device cpu``, seed=0, 10 000 steps): max test accuracy
**0.9827**, final **0.9822** (Google source quotes ≈ 0.9824).

Observed (``--device nntile``, CPU StarPU workers, seed=0): test accuracy
**0.9702** by step 1000 (meets the ≥0.97 floor; full 10 000-step nntile
run is slower on CPU workers than pure torch).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_REPO = Path(__file__).resolve().parents[2]
_TORCH_NNTILE_ROOT = _REPO / "torch_nntile"
if str(_TORCH_NNTILE_ROOT) not in sys.path:
    sys.path.insert(0, str(_TORCH_NNTILE_ROOT))


HIDDEN_WIDTHS = (200, 100, 60, 30)


class LinearBias(nn.Module):
    """Linear + bias via gemm then add (nntile rejects ``nn.Linear`` bias)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, None) + self.bias


class FiveLayerReLU(nn.Module):
    """Bias-aware MLP matching the Google five-layer ReLU tutorial."""

    def __init__(self) -> None:
        super().__init__()
        widths = (28 * 28, *HIDDEN_WIDTHS, 10)
        layers: list[LinearBias] = []
        for in_features, out_features in zip(widths[:-1], widths[1:]):
            layers.append(LinearBias(in_features, out_features))
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
def evaluate_torch(
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
        total_loss += float(
            F.cross_entropy(logits, labels, reduction="sum").item()
        )
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.numel()
    model.train()
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate_nntile(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[float, float]:
    """Evaluate on nntile via forward + host readout (no weight clone)."""
    import torch_nntile
    from torch_nntile.training import cross_entropy

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images_cpu, labels_cpu in loader:
        images = images_cpu.to("nntile")
        labels = labels_cpu.to("nntile")
        logits = model(images)
        loss = cross_entropy(logits, labels, reduction="sum")
        torch_nntile.compile_graph()
        torch_nntile.run()
        torch_nntile.wait()
        logits_cpu = logits.to("cpu")
        loss_cpu = float(loss.to("cpu").item())
        total_loss += loss_cpu
        total_correct += int(
            (logits_cpu.argmax(dim=1) == labels_cpu).sum().item()
        )
        total += labels_cpu.numel()
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
        choices=("cpu", "cuda", "nntile"),
        help="Training device (default: cpu)",
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
    parser.add_argument(
        "--ncpu",
        type=int,
        default=-1,
        help="StarPU CPU workers for nntile (-1 = env default)",
    )
    parser.add_argument(
        "--ncuda",
        type=int,
        default=-1,
        help="StarPU CUDA workers for nntile (-1 = env default)",
    )
    parser.add_argument(
        "--restrict-cuda",
        action="store_true",
        help="Pin nntile kernels to CUDA workers",
    )
    parser.add_argument(
        "--restrict-cpu",
        action="store_true",
        help="Pin nntile kernels to CPU workers",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose StarPU / NNTile context logging",
    )
    parser.add_argument(
        "--skip-accuracy-floor",
        action="store_true",
        help="Do not exit non-zero if final test accuracy < 0.97",
    )
    return parser.parse_args()


def train_torch(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    steps: int,
    train_log_every: int,
    test_every: int,
    device: torch.device,
    n_train: int,
    batch_size: int,
) -> tuple[float, float, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate(0))
    batches = infinite_batches(train_loader)
    max_test_acc = 0.0
    last_test_loss = float("nan")
    last_test_acc = float("nan")
    model.train()

    for step in range(steps + 1):
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

        if step % train_log_every == 0:
            with torch.no_grad():
                train_acc = float(
                    (logits.argmax(dim=1) == labels).float().mean().item()
                )
            print(
                f"{step}: train accuracy={train_acc:.4f} "
                f"loss={float(loss.detach()):.4f} (lr={lr:.6f})"
            )

        if step % test_every == 0:
            test_loss, test_acc = evaluate_torch(model, test_loader, device)
            last_test_loss = test_loss
            last_test_acc = test_acc
            max_test_acc = max(max_test_acc, test_acc)
            epoch = step * batch_size // n_train + 1
            print(
                f"{step}: ********* epoch {epoch} ********* "
                f"test accuracy={test_acc:.4f} test loss={test_loss:.4f}"
            )

    return max_test_acc, last_test_acc, last_test_loss


def train_nntile(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    steps: int,
    train_log_every: int,
    test_every: int,
    n_train: int,
    batch_size: int,
) -> tuple[float, float, float]:
    import torch_nntile
    from torch_nntile.training import Adam, cross_entropy

    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate(0),
    )
    batches = infinite_batches(train_loader)
    max_test_acc = 0.0
    last_test_loss = float("nan")
    last_test_acc = float("nan")
    model.train()

    for step in range(steps + 1):
        lr = learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        images_cpu, labels_cpu = next(batches)
        with torch.no_grad():
            images = images_cpu.to("nntile")
            labels = labels_cpu.to("nntile")

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        torch_nntile.compile_graph()
        torch_nntile.run()
        torch_nntile.wait()

        if step % train_log_every == 0:
            with torch.no_grad():
                logits_cpu = logits.to("cpu")
                loss_cpu = float(loss.to("cpu").item())
                train_acc = float(
                    (logits_cpu.argmax(dim=1) == labels_cpu)
                    .float()
                    .mean()
                    .item()
                )
            print(
                f"{step}: train accuracy={train_acc:.4f} "
                f"loss={loss_cpu:.4f} (lr={lr:.6f})"
            )

        if step % test_every == 0:
            test_loss, test_acc = evaluate_nntile(model, test_loader)
            last_test_loss = test_loss
            last_test_acc = test_acc
            max_test_acc = max(max_test_acc, test_acc)
            epoch = step * batch_size // n_train + 1
            print(
                f"{step}: ********* epoch {epoch} ********* "
                f"test accuracy={test_acc:.4f} test loss={test_loss:.4f}"
            )

    return max_test_acc, last_test_acc, last_test_loss


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    use_nntile = args.device == "nntile"

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

    print(
        "Reproducing Google five-layer ReLU MNIST "
        f"(steps={args.steps}, batch={args.batch_size}, "
        f"device={args.device})"
    )
    print(
        "Source expected test accuracy ≈ 0.9824; "
        "reproduction success floor ≥ 0.97"
    )

    model_cpu = FiveLayerReLU()

    if use_nntile:
        import torch_nntile
        from torch_nntile import _C

        if not _C.has_libnntile():
            raise SystemExit(
                "torch_nntile was built without libnntile. "
                "Set NNTILE_BUILD_DIR and reinstall with "
                "--no-build-isolation --no-deps."
            )
        torch_nntile.init_context(
            ncpu=args.ncpu,
            ncuda=args.ncuda,
            verbose=int(args.verbose),
            cpu_fallback=False,
        )
        if args.restrict_cuda:
            torch_nntile.restrict_cuda()
            print("Worker placement: CUDA only (restrict_cuda)")
        elif args.restrict_cpu:
            torch_nntile.restrict_cpu()
            print("Worker placement: CPU only (restrict_cpu)")
        try:
            with torch.no_grad():
                model = model_cpu.to("nntile")
            del model_cpu
            for param in model.parameters():
                param.requires_grad_(True)
            max_test_acc, last_test_acc, last_test_loss = train_nntile(
                model,
                train_loader,
                test_loader,
                steps=args.steps,
                train_log_every=args.train_log_every,
                test_every=args.test_every,
                n_train=len(train_set),
                batch_size=args.batch_size,
            )
        finally:
            torch_nntile.wait()
            torch_nntile.shutdown_context()
    else:
        device = torch.device(args.device)
        model = model_cpu.to(device)
        max_test_acc, last_test_acc, last_test_loss = train_torch(
            model,
            train_loader,
            test_loader,
            steps=args.steps,
            train_log_every=args.train_log_every,
            test_every=args.test_every,
            device=device,
            n_train=len(train_set),
            batch_size=args.batch_size,
        )

    print(f"max test accuracy: {max_test_acc:.4f}")
    print(
        f"final test accuracy: {last_test_acc:.4f} "
        f"final test loss: {last_test_loss:.4f}"
    )
    if last_test_acc < 0.97 and not args.skip_accuracy_floor:
        raise SystemExit(
            f"Reproduction failed: final test accuracy {last_test_acc:.4f} "
            f"< 0.97 (Google source quotes ≈ 0.9824)"
        )


if __name__ == "__main__":
    main()
