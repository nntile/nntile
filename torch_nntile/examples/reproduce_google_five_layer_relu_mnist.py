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
step). Layers use ``nn.Linear`` with bias (nntile ``aten::linear`` bias path).

Train loss logging is **double-buffered on the host**: after ``run()``
submits the current step (asynchronous), the script prints the previous
step's train metrics so host I/O can overlap StarPU compute, then
``wait()`` synchronizes. Current metrics are snapshotted to host scalars
when needed; the final current loss is printed after the loop. Nntile
step tensors (``logits`` / ``loss``) are ``del``'d every iteration so
``pending_output_reclaim`` can run.

On ``cpu`` / ``cuda`` / ``nntile``, all train/test batches (images + labels)
are moved onto the training device **before** training. The script prints
data-preparation time separately from train/eval compute time.

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
import gc
import math
import sys
import time
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


class FiveLayerReLU(nn.Module):
    """Bias-aware MLP matching the Google five-layer ReLU tutorial."""

    def __init__(self) -> None:
        super().__init__()
        widths = (28 * 28, *HIDDEN_WIDTHS, 10)
        layers: list[nn.Linear] = []
        for in_features, out_features in zip(widths[:-1], widths[1:]):
            layers.append(nn.Linear(in_features, out_features, bias=True))
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


def cycle_batches(batches: list) -> Iterator:
    while True:
        yield from batches


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def materialize_cpu_batches(
    loader: DataLoader,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Eagerly pull one epoch of batches onto contiguous CPU tensors."""
    return [
        (images.contiguous(), labels.contiguous())
        for images, labels in loader
    ]


@torch.no_grad()
def preload_batches_to_device(
    cpu_batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Copy every (images, labels) pair onto ``device`` (cpu or cuda)."""
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for images, labels in cpu_batches:
        out.append(
            (
                images.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )
        )
    synchronize_device(device)
    return out


@torch.no_grad()
def preload_batches_to_nntile(
    cpu_batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Move images/labels to nntile; keep CPU labels for host-side metrics.

    Returns list of ``(images_nntile, labels_nntile, labels_cpu)``.
    """
    out: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for images, labels in cpu_batches:
        out.append((images.to("nntile"), labels.to("nntile"), labels))
    return out


@torch.no_grad()
def evaluate_torch(
    model: nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate on preloaded device batches; return loss, acc, wall seconds."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    t0 = time.perf_counter()
    for images, labels in batches:
        logits = model(images)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        synchronize_device(device)
        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.numel()
    wall_s = time.perf_counter() - t0
    model.train()
    return total_loss / total, total_correct / total, wall_s


@torch.no_grad()
def evaluate_nntile(
    model: nn.Module,
    nntile_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[float, float, float]:
    """Evaluate on preloaded nntile batches; return loss, acc, wall seconds."""
    import torch_nntile
    from torch_nntile.training import cross_entropy

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    t0 = time.perf_counter()
    for images, labels, labels_cpu in nntile_batches:
        logits = model(images)
        loss = cross_entropy(logits, labels, reduction="sum")
        torch_nntile.compile_graph()
        torch_nntile.run()
        torch_nntile.wait()
        with torch.no_grad():
            logits_cpu = logits.to("cpu")
            loss_cpu = float(loss.to("cpu").item())
        del logits
        del loss
        gc.collect()
        total_loss += loss_cpu
        total_correct += int(
            (logits_cpu.argmax(dim=1) == labels_cpu).sum().item()
        )
        total += labels_cpu.numel()
    wall_s = time.perf_counter() - t0
    model.train()
    return total_loss / total, total_correct / total, wall_s


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
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    test_batches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    steps: int,
    train_log_every: int,
    test_every: int,
    device: torch.device,
    n_train: int,
    batch_size: int,
) -> tuple[float, float, float, float, float]:
    """Train on preloaded cpu/cuda batches.

    Returns
    -------
    max_test_acc, last_test_acc, last_test_loss, train_wall_s, eval_wall_s
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate(0))
    batches = cycle_batches(train_batches)
    max_test_acc = 0.0
    last_test_loss = float("nan")
    last_test_acc = float("nan")
    train_step_s = 0.0
    eval_wall_s = 0.0
    # Host-side previous log: (step, accuracy, loss, lr). Printed only after
    # the next step has been ordered.
    pending_log: tuple[int, float, float, float] | None = None
    final_log: tuple[int, float, float, float] | None = None
    model.train()

    t_wall0 = time.perf_counter()
    for step in range(steps + 1):
        lr = learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        images, labels = next(batches)

        t_step0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss_current = F.cross_entropy(logits, labels)
        loss_current.backward()
        optimizer.step()
        synchronize_device(device)

        # Current step ordered: emit previous train log if any.
        if pending_log is not None:
            prev_step, prev_acc, prev_loss, prev_lr = pending_log
            print(
                f"{prev_step}: train accuracy={prev_acc:.4f} "
                f"loss={prev_loss:.4f} (lr={prev_lr:.6f})"
            )
            pending_log = None

        is_last = step == steps
        if step % train_log_every == 0 or is_last:
            with torch.no_grad():
                train_acc = float(
                    (logits.argmax(dim=1) == labels).float().mean().item()
                )
                loss_val = float(loss_current.detach())
            snapshot = (step, train_acc, loss_val, lr)
            if is_last:
                final_log = snapshot
            else:
                pending_log = snapshot
        train_step_s += time.perf_counter() - t_step0

        if step % test_every == 0:
            test_loss, test_acc, eval_s = evaluate_torch(
                model, test_batches, device
            )
            eval_wall_s += eval_s
            last_test_loss = test_loss
            last_test_acc = test_acc
            max_test_acc = max(max_test_acc, test_acc)
            epoch = step * batch_size // n_train + 1
            print(
                f"{step}: ********* epoch {epoch} ********* "
                f"test accuracy={test_acc:.4f} test loss={test_loss:.4f}"
            )

    wall_s = time.perf_counter() - t_wall0

    assert final_log is not None
    _, final_acc, final_loss, final_lr = final_log
    print(
        f"final: train accuracy={final_acc:.4f} "
        f"loss={final_loss:.4f} (lr={final_lr:.6f})"
    )

    print(
        f"timing torch train wall: {wall_s:.3f}s "
        f"(steps+eval+logging over {steps + 1} steps)"
    )
    print(
        f"timing torch train steps: {train_step_s:.3f}s "
        f"({train_step_s / (steps + 1) * 1e3:.2f} ms/step, "
        f"excludes eval)"
    )
    print(f"timing torch eval wall: {eval_wall_s:.3f}s")
    return (
        max_test_acc,
        last_test_acc,
        last_test_loss,
        wall_s,
        eval_wall_s,
    )


def train_nntile(
    model: nn.Module,
    train_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    test_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    steps: int,
    train_log_every: int,
    test_every: int,
    n_train: int,
    batch_size: int,
) -> tuple[float, float, float, float, float]:
    """Train on preloaded nntile batches.

    Returns
    -------
    max_test_acc, last_test_acc, last_test_loss, train_wall_s, eval_wall_s
    """
    import torch_nntile
    from torch_nntile.training import Adam, cross_entropy

    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate(0),
    )
    batches = cycle_batches(train_batches)
    max_test_acc = 0.0
    last_test_loss = float("nan")
    last_test_acc = float("nan")
    train_step_s = 0.0
    eval_wall_s = 0.0
    # Host-side previous log: (step, accuracy, loss, lr). Printed only after
    # the next step has been ordered. Do not keep nntile logits/loss across
    # the next compile_graph() — that blocks pending_output_reclaim.
    pending_log: tuple[int, float, float, float] | None = None
    final_log: tuple[int, float, float, float] | None = None
    model.train()

    t_wall0 = time.perf_counter()
    for step in range(steps + 1):
        lr = learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        images, labels, labels_cpu = next(batches)

        t_step0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss_current = cross_entropy(logits, labels)
        loss_current.backward()
        optimizer.step()
        torch_nntile.compile_graph()
        torch_nntile.run()
        # Current step is submitted asynchronously. Print the previous host
        # log here so it overlaps with StarPU compute; wait() syncs below.
        if pending_log is not None:
            prev_step, prev_acc, prev_loss, prev_lr = pending_log
            print(
                f"{prev_step}: train accuracy={prev_acc:.4f} "
                f"loss={prev_loss:.4f} (lr={prev_lr:.6f})"
            )
            pending_log = None
        torch_nntile.wait()

        is_last = step == steps
        if step % train_log_every == 0 or is_last:
            with torch.no_grad():
                logits_cpu = logits.to("cpu")
                loss_val = float(loss_current.to("cpu").item())
                train_acc = float(
                    (logits_cpu.argmax(dim=1) == labels_cpu)
                    .float()
                    .mean()
                    .item()
                )
            snapshot = (step, train_acc, loss_val, lr)
            if is_last:
                final_log = snapshot
            else:
                pending_log = snapshot

        # Drop step temporaries so mark_output(false) is visible to the
        # pending_output_reclaim pass (end of this run / start of next compile).
        del logits
        del loss_current
        gc.collect()
        train_step_s += time.perf_counter() - t_step0

        if step % test_every == 0:
            test_loss, test_acc, eval_s = evaluate_nntile(
                model, test_batches
            )
            eval_wall_s += eval_s
            last_test_loss = test_loss
            last_test_acc = test_acc
            max_test_acc = max(max_test_acc, test_acc)
            epoch = step * batch_size // n_train + 1
            print(
                f"{step}: ********* epoch {epoch} ********* "
                f"test accuracy={test_acc:.4f} test loss={test_loss:.4f}"
            )

    wall_s = time.perf_counter() - t_wall0

    assert final_log is not None
    _, final_acc, final_loss, final_lr = final_log
    print(
        f"final: train accuracy={final_acc:.4f} "
        f"loss={final_loss:.4f} (lr={final_lr:.6f})"
    )

    print(
        f"timing nntile train wall: {wall_s:.3f}s "
        f"(steps+eval+logging over {steps + 1} steps)"
    )
    print(
        f"timing nntile train steps: {train_step_s:.3f}s "
        f"({train_step_s / (steps + 1) * 1e3:.2f} ms/step, "
        f"includes compile/run/wait, host readout, gc; excludes eval)"
    )
    print(f"timing nntile eval wall: {eval_wall_s:.3f}s")
    torch_nntile.print_info()
    return (
        max_test_acc,
        last_test_acc,
        last_test_loss,
        wall_s,
        eval_wall_s,
    )


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

    print("Materializing CPU train/test batches...")
    t_mat0 = time.perf_counter()
    train_cpu = materialize_cpu_batches(train_loader)
    test_cpu = materialize_cpu_batches(test_loader)
    materialize_s = time.perf_counter() - t_mat0
    print(
        f"timing CPU materialize: {materialize_s:.3f}s "
        f"({len(train_cpu)} train batches, {len(test_cpu)} test batches)"
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
            print("Preloading train/test batches to nntile...")
            t_pre0 = time.perf_counter()
            with torch.no_grad():
                train_nnt = preload_batches_to_nntile(train_cpu)
                test_nnt = preload_batches_to_nntile(test_cpu)
                model = model_cpu.to("nntile")
            # Ensure transfers are complete before stopping the clock.
            torch_nntile.wait()
            preprocess_s = time.perf_counter() - t_pre0
            n_train_elems = sum(x.numel() for x, _ in train_cpu)
            n_test_elems = sum(x.numel() for x, _ in test_cpu)
            print(
                f"timing host→nntile preprocess: {preprocess_s:.3f}s "
                f"(train images {n_train_elems}, "
                f"test images {n_test_elems}, + labels + model)"
            )
            del model_cpu
            del train_cpu
            del test_cpu
            for param in model.parameters():
                param.requires_grad_(True)
            max_test_acc, last_test_acc, last_test_loss, _, _ = train_nntile(
                model,
                train_nnt,
                test_nnt,
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
        print(f"Preloading train/test batches to {device}...")
        t_pre0 = time.perf_counter()
        with torch.no_grad():
            train_dev = preload_batches_to_device(train_cpu, device)
            test_dev = preload_batches_to_device(test_cpu, device)
            model = model_cpu.to(device)
            synchronize_device(device)
        preprocess_s = time.perf_counter() - t_pre0
        n_train_elems = sum(x.numel() for x, _ in train_cpu)
        n_test_elems = sum(x.numel() for x, _ in test_cpu)
        print(
            f"timing host→{device} preprocess: {preprocess_s:.3f}s "
            f"(train images {n_train_elems}, "
            f"test images {n_test_elems}, + labels + model)"
        )
        del model_cpu
        del train_cpu
        del test_cpu
        max_test_acc, last_test_acc, last_test_loss, _, _ = train_torch(
            model,
            train_dev,
            test_dev,
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
