#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_deep_relu_mnist.py
# Train DeepReLU on the full MNIST training set (60k batch) on nntile.

"""Full-batch MNIST training with DeepReLU on device=\"nntile\".

By default this script trains only on nntile (useful for larger tiled /
multi-worker runs). Pass ``--compare-torch`` to also train a CPU PyTorch
reference and print per-epoch loss / final weight parity.

Cross-entropy is evaluated on nntile via ``torch_nntile.training.cross_entropy``
(same tensor-op chain as ``NNCrossEntropyOp`` in libnntile). Logits are
``[batch, classes]`` (class dim last). The scalar loss lives on
``device="nntile"``; read it with ``loss.to("cpu")`` after ``compile_graph()`` and
``run()`` in graph mode.

StarPU may use CUDA workers for the nntile path via ``STARPU_NCUDA`` and
``--restrict-cuda``.

Axis-group naming and tiling (optional) are configured in this script:

- ``batch`` — input/logits batch dimension
- ``features`` — flattened image dimension (784)
- ``hidden`` — hidden MLP width (``--hidden-dim``); named on each linear
  weight/grad/velocity matrix row or column of that size
- ``classes`` — output logits dimension (10)

Nntile-only (tiled CUDA workers)::

    export LD_LIBRARY_PATH=$PWD/build/nntile:/opt/starpu/lib
    STARPU_NCPU=0 STARPU_NCUDA=2 \\
    python torch_nntile/examples/train_deep_relu_mnist.py \\
        --restrict-cuda \\
        --epochs 5 \\
        --axis-tiling batch=15000,15000,15000,15000 \\
        --axis-tiling features=392,392 \\
        --axis-tiling hidden=128,128

CPU torch parity check::

    STARPU_NCPU=4 STARPU_NCUDA=0 \\
    python torch_nntile/examples/train_deep_relu_mnist.py \\
        --compare-torch --epochs 5

Run instructions and expected output:
``docs/torch_nntile.md`` (DeepReLU MNIST example section).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision import datasets

_REPO = Path(__file__).resolve().parents[2]
_TORCH_NNTILE_ROOT = _REPO / "torch_nntile"
if str(_TORCH_NNTILE_ROOT) not in sys.path:
    sys.path.insert(0, str(_TORCH_NNTILE_ROOT))


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


def parse_axis_tiling_arg(spec: str) -> tuple[str, list[int]]:
    """Parse ``name=size`` or ``name=size,size,...``."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"axis tiling must be NAME=SIZES, got {spec!r}"
        )
    name, sizes_text = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("axis group name must be non-empty")
    sizes: list[int] = []
    for part in sizes_text.split(","):
        part = part.strip()
        if not part:
            raise argparse.ArgumentTypeError(
                f"invalid tile size list in {spec!r}"
            )
        value = int(part)
        if value <= 0:
            raise argparse.ArgumentTypeError("tile sizes must be positive")
        sizes.append(value)
    return name, sizes


def build_axis_group_tiling(
    specs: list[str],
) -> dict[str, list[int]]:
    tiling: dict[str, list[int]] = {}
    for spec in specs:
        name, sizes = parse_axis_tiling_arg(spec)
        tiling[name] = sizes
    return tiling


def _clone_state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def build_torch_model(
    seed: int,
    hidden_dim: int,
    depth: int,
) -> torch.nn.Module:
    """Build DeepReLU on CPU with Kaiming init."""
    from torch_nntile.models import DeepReLU

    torch.manual_seed(seed)
    model = DeepReLU.mnist(hidden_dim=hidden_dim, depth=depth)
    model.init_kaiming_uniform_(seed=seed)
    return model


def train_torch_reference(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
) -> list[float]:
    """Pure-PyTorch full-batch SGD on CPU."""
    losses: list[float] = []
    for epoch in range(epochs):
        for param in model.parameters():
            param.grad = None
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param.add_(param.grad, alpha=-learning_rate)
        value = float(loss.detach().item())
        losses.append(value)
        print(f"[cpu] epoch {epoch + 1}/{epochs}  loss={value:.6f}")
    return losses


def _name_hidden_on_matrix(tensor: torch.Tensor, hidden_dim: int) -> None:
    """Tag matrix rows/cols of size ``hidden_dim`` with the ``hidden`` axis group."""
    import torch_nntile

    if tensor.ndim != 2:
        return
    out_features, in_features = tensor.shape
    names: dict[int, str] = {}
    if out_features == hidden_dim:
        names[0] = "hidden"
    if in_features == hidden_dim:
        names[1] = "hidden"
    if names:
        torch_nntile.set_axis_group_name(tensor, names)


def name_mnist_axis_groups(
    model: torch.nn.Module,
    x: torch.Tensor,
    logits: torch.Tensor,
    *,
    hidden_dim: int,
) -> None:
    """Name axis groups for DeepReLU MNIST training on nntile."""
    import torch_nntile

    torch_nntile.set_axis_group_name(x, {0: "batch", 1: "features"})
    torch_nntile.set_axis_group_name(logits, {1: "classes"})
    for module in model.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        _name_hidden_on_matrix(module.weight, hidden_dim)
        if module.weight.grad is not None:
            _name_hidden_on_matrix(module.weight.grad, hidden_dim)
    optimizer = getattr(model, "_nntile_optimizer", None)
    if optimizer is not None:
        for velocity in optimizer._velocity.values():
            _name_hidden_on_matrix(velocity, hidden_dim)


def build_nntile_model(
    hidden_dim: int,
    depth: int,
    state_dict: dict[str, torch.Tensor],
) -> torch.nn.Module:
    """Clone CPU ``state_dict`` onto ``device='nntile'`` (requires init_context)."""
    from torch_nntile.models import DeepReLU

    model = DeepReLU.mnist(hidden_dim=hidden_dim, depth=depth)
    model.load_state_dict(state_dict)
    return model.to("nntile")


def train_on_nntile(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    axis_group_tiling: dict[str, list[int]] | None = None,
    print_axis_groups: bool = False,
) -> list[float]:
    import torch_nntile
    from torch_nntile.training import train_full_batch_step

    x = images.to("nntile")
    y = labels.to("nntile")
    if axis_group_tiling is not None:
        for name, tile_sizes in axis_group_tiling.items():
            torch_nntile.set_axis_group_tiling(name, tile_sizes)

    def name_axis_groups(x: torch.Tensor, logits: torch.Tensor) -> None:
        name_mnist_axis_groups(model, x, logits, hidden_dim=hidden_dim)

    losses: list[float] = []
    for epoch in range(epochs):
        loss = train_full_batch_step(
            model,
            x,
            y,
            learning_rate,
            name_axis_groups=name_axis_groups,
            axis_group_tiling=axis_group_tiling,
            print_axis_groups=print_axis_groups and epoch == 0,
        )
        losses.append(loss)
        print(f"[nntile] epoch {epoch + 1}/{epochs}  loss={loss:.6f}")
    return losses


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/tmp/mnist_torch_nntile")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument(
        "--compare-torch",
        action="store_true",
        help=(
            "Also train a CPU PyTorch reference and print loss/weight "
            "parity (default: nntile-only)"
        ),
    )
    parser.add_argument(
        "--axis-tiling",
        action="append",
        default=[],
        metavar="NAME=SIZES",
        help=(
            "Axis-group tiling for nntile, e.g. batch=15000,15000,15000,15000 "
            "or features=392,392 or hidden=128,128. Repeat for multiple groups."
        ),
    )
    parser.add_argument(
        "--print-axis-groups",
        action="store_true",
        help="Print axis groups after the first nntile training step",
    )
    parser.add_argument(
        "--restrict-cuda",
        action="store_true",
        help="Pin nntile kernels to CUDA workers (requires ncuda > 0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose StarPU / NNTile context logging",
    )
    parser.add_argument("--output-dir", default="deep_relu_mnist_runs")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    axis_group_tiling = build_axis_group_tiling(args.axis_tiling)
    compare_torch = bool(args.compare_torch)

    if compare_torch:
        print("Mode: nntile + CPU torch parity")
    else:
        print("Mode: nntile-only (pass --compare-torch for CPU parity)")
    if axis_group_tiling:
        print(f"Axis-group tiling: {axis_group_tiling}")
    print(f"DeepReLU hidden_dim={args.hidden_dim} depth={args.depth}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MNIST training set (60 000 images, single batch)...")
    images, labels = load_mnist_full_batch(args.data_dir)
    print(f"  images {tuple(images.shape)}, labels {tuple(labels.shape)}")

    import torch_nntile
    from torch_nntile import _C
    from torch_nntile.training import clone_model_weights, max_weight_delta

    if not _C.has_libnntile():
        raise SystemExit(
            "torch_nntile was built without libnntile. "
            "Set NNTILE_BUILD_DIR and reinstall."
        )

    model_init = build_torch_model(
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    )
    init_weights = _clone_state_dict_cpu(model_init)

    torch_losses: list[float] | None = None
    final_torch: dict[str, torch.Tensor] | None = None
    if compare_torch:
        print("\nTraining on torch (cpu)...")
        torch_losses = train_torch_reference(
            model_init,
            images,
            labels,
            epochs=args.epochs,
            learning_rate=args.lr,
        )
        final_torch = _clone_state_dict_cpu(model_init)
    del model_init

    torch_nntile.init_context(
        ncpu=-1,
        ncuda=-1,
        verbose=int(args.verbose),
        cpu_fallback=False,
    )
    if args.restrict_cuda:
        torch_nntile.restrict_cuda()
        print("Worker placement: CUDA only (restrict_cuda)")

    try:
        model_nnt = build_nntile_model(
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            state_dict=init_weights,
        )
        # Do not .cpu() / clone_model_weights before the first tiled compile:
        # that seals untiled layouts into the TileGraph and later
        # --axis-tiling hits layout_fingerprint mismatch.
        print(
            "Initial weights loaded from torch state_dict onto nntile "
            "(host round-trip deferred until after training)"
        )

        print("\nTraining on nntile...")
        nnt_losses = train_on_nntile(
            model_nnt,
            images,
            labels,
            epochs=args.epochs,
            learning_rate=args.lr,
            hidden_dim=args.hidden_dim,
            axis_group_tiling=axis_group_tiling or None,
            print_axis_groups=args.print_axis_groups,
        )

        nnt_path = output_dir / "deep_relu_mnist_nntile.pt"
        # Host gather only after training (safe with --axis-tiling).
        final_nnt = clone_model_weights(model_nnt)
        torch.save(final_nnt, nnt_path)
        print(f"\nSaved nntile model (CPU tensors) to {nnt_path}")

        if compare_torch:
            assert torch_losses is not None and final_torch is not None
            torch_path = output_dir / "deep_relu_mnist_torch_cpu.pt"
            torch.save(final_torch, torch_path)
            weight_delta = max_weight_delta(final_torch, final_nnt)

            print("\nLoss comparison (torch/cpu vs nntile):")
            for epoch, (loss_torch, loss_nnt) in enumerate(
                zip(torch_losses, nnt_losses), start=1
            ):
                print(
                    f"  epoch {epoch}: torch={loss_torch:.6f}  "
                    f"nntile={loss_nnt:.6f}  "
                    f"diff={abs(loss_torch - loss_nnt):.3e}"
                )

            print(
                f"\nFinal weight max |torch - nntile| = {weight_delta:.3e}"
            )
            print(f"Saved torch model (CPU tensors) to {torch_path}")
        else:
            print("\nNntile losses:")
            for epoch, loss_nnt in enumerate(nnt_losses, start=1):
                print(f"  epoch {epoch}: nntile={loss_nnt:.6f}")
    finally:
        torch_nntile.wait()
        torch_nntile.shutdown_context()


if __name__ == "__main__":
    main()
