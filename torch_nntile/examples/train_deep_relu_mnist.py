#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_deep_relu_mnist.py
# Train DeepReLU on the full MNIST training set (60k batch, optional grad accum).

"""Full-batch MNIST training with DeepReLU (nntile parallelization demo).

Trains ``DeepReLU.mnist()`` on the entire MNIST training set as one batch
(60 000 images). Primary goal: show how well nntile parallelizes a large
DeepReLU across StarPU CPU/CUDA workers.

Supports ``--device cpu`` / ``cuda`` (PyTorch SGD) and ``--device nntile``
(StarPU). Pass ``--ncpu`` / ``--ncuda`` to set StarPU workers for the
nntile path (``-1`` = env default).

Optional ``--grad-accum-steps N`` splits the 60 000-image batch into ``N``
equal microbatches, scales each microbatch loss by ``1/N``, accumulates
gradients, then takes one SGD step per epoch (same effective update as
full-batch when ``N`` divides 60 000). Default ``N=1`` is full-batch.
On nntile, each microbatch compiles/runs before the next so activations
can reclaim; grads stay live until the step.

Torch cannot use CUDA and the PrivateUse1 ``nntile`` device in one process
(PyTorch >= 2.8). This script imports ``torch_nntile`` only for
``--device nntile``; CUDA/CPU runs load ``DeepReLU`` without registering
nntile. Use separate processes to compare cuda vs nntile.

Before training, microbatches (images + labels) and the model are moved
onto the training device; the script prints prefetch time and wall
training time.

Pass ``--compare-torch`` with ``--device nntile`` to also train a CPU
PyTorch reference and print per-epoch loss / final weight parity.
Nntile-only flags (``--ncpu``, ``--ncuda``, ``--restrict-*``,
``--axis-tiling``, ``--print-axis-groups``, ``--compare-torch``) are
accepted on ``--device cpu`` / ``cuda`` but ignored (reported in output).

Axis-group naming and tiling (optional) are configured in this script:

- ``batch`` - input/logits batch dimension
- ``features`` - flattened image dimension (784)
- ``hidden`` - hidden MLP width (``--hidden-dim``); named on each linear
  weight/grad/velocity matrix row or column of that size
- ``classes`` - output logits dimension (10)

Nntile (tiled CUDA workers)::

    export LD_LIBRARY_PATH=$PWD/build/nntile:/opt/starpu/lib
    python torch_nntile/examples/train_deep_relu_mnist.py \\
        --device nntile --ncpu 0 --ncuda 2 --restrict-cuda \\
        --epochs 5 \\
        --axis-tiling batch=15000,15000,15000,15000 \\
        --axis-tiling features=392,392 \\
        --axis-tiling hidden=128,128

Gradient accumulation (4 microbatches of 15 000; lower peak activation memory)::

    python torch_nntile/examples/train_deep_relu_mnist.py \\
        --device nntile --ncpu 0 --ncuda 1 --restrict-cuda \\
        --grad-accum-steps 4 --epochs 5

CPU torch reference::

    python torch_nntile/examples/train_deep_relu_mnist.py \\
        --device cpu --epochs 5

CUDA torch reference (separate process from any nntile run)::

    python torch_nntile/examples/train_deep_relu_mnist.py \\
        --device cuda --epochs 5

Nntile + CPU torch parity::

    python torch_nntile/examples/train_deep_relu_mnist.py \\
        --device nntile --ncpu 4 --ncuda 0 --compare-torch --epochs 5

Run instructions and expected output:
``docs/torch_nntile.md`` (DeepReLU MNIST example section).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Type

import torch
import torch.nn as nn
from torchvision import datasets

_REPO = Path(__file__).resolve().parents[2]
_TORCH_NNTILE_ROOT = _REPO / "torch_nntile"
if str(_TORCH_NNTILE_ROOT) not in sys.path:
    sys.path.insert(0, str(_TORCH_NNTILE_ROOT))

_DeepReLU: Type[nn.Module] | None = None


def _deep_relu_class() -> Type[nn.Module]:
    """Load DeepReLU without importing ``torch_nntile`` (no PrivateUse1)."""
    global _DeepReLU
    if _DeepReLU is not None:
        return _DeepReLU
    path = _TORCH_NNTILE_ROOT / "torch_nntile" / "models" / "deep_relu.py"
    spec = importlib.util.spec_from_file_location(
        "torch_nntile_deep_relu_standalone",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load DeepReLU from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _DeepReLU = module.DeepReLU
    return _DeepReLU


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


def split_microbatches(
    images: torch.Tensor,
    labels: torch.Tensor,
    grad_accum_steps: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Split a full batch into ``grad_accum_steps`` equal microbatches.

    Must run on host (or cuda/cpu) tensors: nntile ``narrow``/``chunk`` are
    float32-only, while MNIST labels are int64.
    """
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    n = int(images.shape[0])
    if n != int(labels.shape[0]):
        raise ValueError(
            f"images/labels length mismatch: {n} vs {int(labels.shape[0])}"
        )
    if n % grad_accum_steps != 0:
        raise ValueError(
            f"batch size {n} must be divisible by "
            f"grad_accum_steps={grad_accum_steps}"
        )
    microbatch = n // grad_accum_steps
    return [
        (
            images[i * microbatch : (i + 1) * microbatch],
            labels[i * microbatch : (i + 1) * microbatch],
        )
        for i in range(grad_accum_steps)
    ]


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


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clone_state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def print_tensor_norm(label: str, tensor: torch.Tensor) -> None:
    """Print ``tensor.norm()`` without recording autograd (norm has no nntile backward)."""
    with torch.no_grad():
        print(f"{label}: {tensor.norm().detach().cpu()}")


def print_state_dict_norms(
    prefix: str,
    state: dict[str, torch.Tensor],
) -> None:
    """Print Frobenius norms of state-dict tensors under ``torch.no_grad()``."""
    for name, tensor in state.items():
        print_tensor_norm(f"{prefix} {name} norm", tensor)


def build_torch_model(
    seed: int,
    hidden_dim: int,
    depth: int,
) -> torch.nn.Module:
    """Build DeepReLU on CPU with Kaiming init (no torch_nntile import)."""
    DeepReLU = _deep_relu_class()
    torch.manual_seed(seed)
    model = DeepReLU.mnist(hidden_dim=hidden_dim, depth=depth)
    model.init_kaiming_uniform_(seed=seed)
    return model


def train_torch_reference(
    model: torch.nn.Module,
    microbatches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    epochs: int,
    learning_rate: float,
    device: torch.device | None = None,
) -> list[float]:
    """Pure-PyTorch SGD with optional gradient accumulation over microbatches."""
    n_accum = len(microbatches)
    if n_accum < 1:
        raise ValueError("microbatches must be non-empty")
    scale = 1.0 / n_accum
    losses: list[float] = []
    for epoch in range(epochs):
        for param in model.parameters():
            param.grad = None
        loss_sum = 0.0
        for images, labels in microbatches:
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if n_accum == 1:
                loss.backward()
            else:
                loss.backward(
                    gradient=torch.tensor(
                        scale, dtype=loss.dtype, device=loss.device
                    )
                )
            loss_sum += float(loss.detach().item())
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param.add_(param.grad, alpha=-learning_rate)
        if device is not None:
            synchronize_device(device)
        value = loss_sum / n_accum
        losses.append(value)
        print(f"[torch] epoch {epoch + 1}/{epochs}  loss={value:.6f}")
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


def name_prefetched_mnist_axis_groups(
    model: torch.nn.Module,
    microbatches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    hidden_dim: int,
) -> None:
    """Name axis groups on every prefetched microbatch + model weights.

    The first ``compile_graph`` seals *all* pending ingress scatters (every
    prefetched microbatch), not only the microbatch being trained. Inputs that
    are still unnamed are lowered untiled; naming them later with
    ``--axis-tiling`` then raises ``layout_fingerprint mismatch``. Call this
    before the first tiled compile when ``len(microbatches) > 1``.
    """
    import torch_nntile

    for images, labels in microbatches:
        torch_nntile.set_axis_group_name(images, {0: "batch", 1: "features"})
        torch_nntile.set_axis_group_name(labels, {0: "batch"})
    for module in model.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        _name_hidden_on_matrix(module.weight, hidden_dim)


def name_mnist_axis_groups(
    model: torch.nn.Module,
    x: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    *,
    hidden_dim: int,
) -> None:
    """Name axis groups for DeepReLU MNIST training on nntile."""
    import torch_nntile

    torch_nntile.set_axis_group_name(x, {0: "batch", 1: "features"})
    torch_nntile.set_axis_group_name(labels, {0: "batch"})
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
    """Clone CPU ``state_dict`` onto ``device='nntile'`` (requires init_context).

    Call only after ``import torch_nntile`` has registered the device.
    """
    DeepReLU = _deep_relu_class()
    model = DeepReLU.mnist(hidden_dim=hidden_dim, depth=depth)
    with torch.no_grad():
        model.load_state_dict(state_dict)
        model = model.to("nntile")
    return model


def train_on_nntile(
    model: torch.nn.Module,
    microbatches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    axis_group_tiling: dict[str, list[int]] | None = None,
    print_axis_groups: bool = False,
) -> list[float]:
    """Train on preloaded nntile microbatches with gradient accumulation.

    Each microbatch records forward/backward, then ``compile_graph`` /
    ``run`` so activations can ``INVALIDATE`` before the next microbatch.
    Gradients stay on ``param.grad`` until the last microbatch's SGD step;
    ``zero_grad`` runs before that final compile (TensorRef reclaim pattern).

    Axis-group naming must run before ``set_axis_group_tiling`` on every
    microbatch (tiling is applied at ``compile_graph``; unknown names fail).
    With multiple prefetched microbatches, name *all* of them before the
    first tiled compile - that compile also seals the other microbatches'
    pending ingress scatters.
    """
    import torch_nntile
    from torch_nntile.training import SGD, cross_entropy

    n_accum = len(microbatches)
    if n_accum < 1:
        raise ValueError("microbatches must be non-empty")
    scale = 1.0 / n_accum

    optimizer = getattr(model, "_nntile_optimizer", None)
    if optimizer is None or optimizer.param_groups[0]["lr"] != learning_rate:
        optimizer = SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
        )
        model._nntile_optimizer = optimizer

    # Prefetch records ingress scatters for every microbatch; the first
    # compile_graph seals them all. Name + tile before that seal.
    if axis_group_tiling is not None:
        name_prefetched_mnist_axis_groups(
            model, microbatches, hidden_dim=hidden_dim
        )

    losses: list[float] = []
    for epoch in range(epochs):
        loss_sum = 0.0
        for mb_idx, (images, labels) in enumerate(microbatches):
            is_last = mb_idx == n_accum - 1
            logits = model(images)
            loss = cross_entropy(logits, labels)
            if n_accum == 1:
                loss.backward()
            else:
                # Prefer gradient= over ``loss * scale``: Python floats can
                # become CPU 0-dim tensors and hit mul.Tensor (both must be
                # nntile). CE folds a constant scalar grad_output.
                loss.backward(
                    gradient=torch.tensor(scale, dtype=torch.float32).to(
                        "nntile"
                    )
                )

            if is_last:
                optimizer.step()

            # Name -> tile -> compile (same order as train_full_batch_step).
            # New microbatch inputs/logits need names each time; weight/grad
            # names are reapplied safely. Do this while logits are alive.
            name_mnist_axis_groups(
                model, images, labels, logits, hidden_dim=hidden_dim
            )
            if axis_group_tiling is not None:
                for name, tile_sizes in axis_group_tiling.items():
                    torch_nntile.set_axis_group_tiling(name, tile_sizes)
            if print_axis_groups and epoch == 0 and is_last:
                torch_nntile.print_axis_groups()

            # Detached scalar for host readout; drop activations before
            # compile so INVALIDATEs share this sealed phase. Keep grads
            # until the last microbatch (then zero before compile).
            step_loss = loss.detach()
            del logits
            del loss
            if is_last:
                optimizer.zero_grad(set_to_none=True)

            torch_nntile.compile_graph()
            torch_nntile.run()
            with torch.no_grad():
                loss_cpu = step_loss.to("cpu")
            loss_sum += float(loss_cpu.item())
            del step_loss

        value = loss_sum / n_accum
        losses.append(value)
        print(f"[nntile] epoch {epoch + 1}/{epochs}  loss={value:.6f}")
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
        "--grad-accum-steps",
        type=int,
        default=1,
        help=(
            "Split the 60k batch into N equal microbatches, accumulate "
            "grads (loss scaled by 1/N), one SGD step per epoch "
            "(default: 1 = full batch; N must divide 60000)"
        ),
    )
    parser.add_argument(
        "--device",
        default="nntile",
        choices=("cpu", "cuda", "nntile"),
        help="Training device (default: nntile)",
    )
    parser.add_argument(
        "--ncpu",
        type=int,
        default=-1,
        help=(
            "StarPU CPU workers for nntile (-1 = env default; "
            "ignored on --device cpu/cuda)"
        ),
    )
    parser.add_argument(
        "--ncuda",
        type=int,
        default=-1,
        help=(
            "StarPU CUDA workers for nntile (-1 = env default; "
            "ignored on --device cpu/cuda)"
        ),
    )
    parser.add_argument(
        "--compare-torch",
        action="store_true",
        help=(
            "With --device nntile: also train a CPU PyTorch reference and "
            "print loss/weight parity (ignored on --device cpu/cuda)"
        ),
    )
    parser.add_argument(
        "--axis-tiling",
        action="append",
        default=[],
        metavar="NAME=SIZES",
        help=(
            "Axis-group tiling for nntile, e.g. batch=15000,15000,15000,15000 "
            "or features=392,392 or hidden=128,128. Repeat for multiple "
            "groups (ignored on --device cpu/cuda)."
        ),
    )
    parser.add_argument(
        "--print-axis-groups",
        action="store_true",
        help=(
            "Print axis groups after the first nntile training step "
            "(ignored on --device cpu/cuda)"
        ),
    )
    parser.add_argument(
        "--restrict-cuda",
        action="store_true",
        help=(
            "Pin nntile kernels to CUDA workers (requires ncuda > 0; "
            "ignored on --device cpu/cuda)"
        ),
    )
    parser.add_argument(
        "--restrict-cpu",
        action="store_true",
        help=(
            "Pin nntile kernels to CPU workers "
            "(ignored on --device cpu/cuda)"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Verbose StarPU / NNTile context logging, and print weight "
            "norms under torch.no_grad()"
        ),
    )
    parser.add_argument("--output-dir", default="deep_relu_mnist_runs")
    return parser


def _nntile_only_args_set(args: argparse.Namespace) -> list[str]:
    """Return nntile-only CLI flags that were explicitly set."""
    ignored: list[str] = []
    if args.ncpu != -1:
        ignored.append(f"--ncpu={args.ncpu}")
    if args.ncuda != -1:
        ignored.append(f"--ncuda={args.ncuda}")
    if args.restrict_cuda:
        ignored.append("--restrict-cuda")
    if args.restrict_cpu:
        ignored.append("--restrict-cpu")
    if args.axis_tiling:
        ignored.append("--axis-tiling")
    if args.print_axis_groups:
        ignored.append("--print-axis-groups")
    if args.compare_torch:
        ignored.append("--compare-torch")
    return ignored


def main() -> None:
    args = _build_parser().parse_args()
    axis_group_tiling = build_axis_group_tiling(args.axis_tiling)
    compare_torch = bool(args.compare_torch)
    use_nntile = args.device == "nntile"

    if args.grad_accum_steps < 1:
        raise SystemExit("--grad-accum-steps must be >= 1")

    if use_nntile and args.restrict_cuda and args.restrict_cpu:
        raise SystemExit("Pass only one of --restrict-cuda / --restrict-cpu")

    if use_nntile and compare_torch:
        print("Mode: nntile + CPU torch parity")
    else:
        print(f"Mode: {args.device}-only")

    if use_nntile:
        print(f"StarPU workers: ncpu={args.ncpu} ncuda={args.ncuda}")
        if axis_group_tiling:
            print(f"Axis-group tiling: {axis_group_tiling}")
    else:
        # Accept nntile-only flags on torch paths; report and ignore them.
        ignored = _nntile_only_args_set(args)
        if ignored:
            print(
                "Ignoring nntile-only arguments on "
                f"--device {args.device}: {', '.join(ignored)}"
            )
        compare_torch = False
        axis_group_tiling = {}

    print(
        f"DeepReLU hidden_dim={args.hidden_dim} depth={args.depth} "
        f"epochs={args.epochs} device={args.device} "
        f"grad_accum_steps={args.grad_accum_steps}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MNIST training set (60 000 images, single batch)...")
    images_cpu, labels_cpu = load_mnist_full_batch(args.data_dir)
    print(
        f"  images {tuple(images_cpu.shape)}, "
        f"labels {tuple(labels_cpu.shape)}"
    )
    try:
        cpu_microbatches = split_microbatches(
            images_cpu, labels_cpu, args.grad_accum_steps
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    microbatch_size = int(cpu_microbatches[0][0].shape[0])
    print(
        f"  grad accum: {args.grad_accum_steps} microbatches "
        f"x {microbatch_size} samples"
    )

    model_init = build_torch_model(
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    )
    init_weights = _clone_state_dict_cpu(model_init)
    if args.verbose:
        print_state_dict_norms("init", init_weights)

    torch_losses: list[float] | None = None
    final_torch: dict[str, torch.Tensor] | None = None
    if compare_torch:
        print("\nTraining CPU torch reference (parity)...")
        torch_losses = train_torch_reference(
            model_init,
            cpu_microbatches,
            epochs=args.epochs,
            learning_rate=args.lr,
            device=torch.device("cpu"),
        )
        final_torch = _clone_state_dict_cpu(model_init)
        if args.verbose:
            print_state_dict_norms("torch/final", final_torch)
    del model_init

    if use_nntile:
        import torch_nntile
        from torch_nntile.training import clone_model_weights, max_weight_delta

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
            print("Prefetching MNIST + model to nntile...")
            t_pre0 = time.perf_counter()
            with torch.no_grad():
                microbatches = [
                    (images.to("nntile"), labels.to("nntile"))
                    for images, labels in cpu_microbatches
                ]
                model_nnt = build_nntile_model(
                    hidden_dim=args.hidden_dim,
                    depth=args.depth,
                    state_dict=init_weights,
                )
            # Ensure transfers are complete before stopping the clock.
            torch_nntile.wait()
            prefetch_s = time.perf_counter() - t_pre0
            n_image_elems = sum(img.numel() for img, _ in cpu_microbatches)
            n_label_elems = sum(lab.numel() for _, lab in cpu_microbatches)
            print(
                f"timing host->nntile prefetch: {prefetch_s:.3f}s "
                f"(MNIST images {n_image_elems}, "
                f"labels {n_label_elems}, + model)"
            )
            # Do not .cpu() / clone_model_weights before the first tiled
            # compile: that seals untiled layouts into the TileGraph and
            # later --axis-tiling hits layout_fingerprint mismatch.
            # With --grad-accum-steps > 1, train_on_nntile also names every
            # prefetched microbatch before that first tiled compile (pending
            # ingress scatters are sealed together).

            print("\nTraining on nntile...")
            t_train0 = time.perf_counter()
            nnt_losses = train_on_nntile(
                model_nnt,
                microbatches,
                epochs=args.epochs,
                learning_rate=args.lr,
                hidden_dim=args.hidden_dim,
                axis_group_tiling=axis_group_tiling or None,
                print_axis_groups=args.print_axis_groups,
            )
            train_wall_s = time.perf_counter() - t_train0
            print(
                f"timing nntile train wall: {train_wall_s:.3f}s "
                f"({args.epochs} epochs, "
                f"{args.grad_accum_steps} accum steps/epoch)"
            )

            nnt_path = output_dir / "deep_relu_mnist_nntile.pt"
            # Host gather only after training (safe with --axis-tiling).
            final_nnt = clone_model_weights(model_nnt)
            torch.save(final_nnt, nnt_path)
            print(f"\nSaved nntile model (CPU tensors) to {nnt_path}")
            if args.verbose:
                print_state_dict_norms("nntile/final", final_nnt)

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
                    f"\nFinal weight max |torch - nntile| = "
                    f"{weight_delta:.3e}"
                )
                print(f"Saved torch model (CPU tensors) to {torch_path}")
        finally:
            torch_nntile.wait()
            torch_nntile.shutdown_context()
    else:
        # Do not import torch_nntile here: PrivateUse1 registration breaks
        # CUDA autograd in the same process (PyTorch >= 2.8).
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA is not available")

        DeepReLU = _deep_relu_class()
        print(f"Prefetching MNIST + model to {device}...")
        t_pre0 = time.perf_counter()
        with torch.no_grad():
            microbatches = [
                (
                    images.to(device, non_blocking=True),
                    labels.to(device, non_blocking=True),
                )
                for images, labels in cpu_microbatches
            ]
            model = DeepReLU.mnist(
                hidden_dim=args.hidden_dim,
                depth=args.depth,
            )
            model.load_state_dict(init_weights)
            model = model.to(device)
            synchronize_device(device)
        prefetch_s = time.perf_counter() - t_pre0
        n_image_elems = sum(img.numel() for img, _ in cpu_microbatches)
        n_label_elems = sum(lab.numel() for _, lab in cpu_microbatches)
        print(
            f"timing host->{device} prefetch: {prefetch_s:.3f}s "
            f"(MNIST images {n_image_elems}, "
            f"labels {n_label_elems}, + model)"
        )

        print(f"\nTraining on torch ({device})...")
        t_train0 = time.perf_counter()
        train_torch_reference(
            model,
            microbatches,
            epochs=args.epochs,
            learning_rate=args.lr,
            device=device,
        )
        train_wall_s = time.perf_counter() - t_train0
        print(
            f"timing torch train wall: {train_wall_s:.3f}s "
            f"({args.epochs} epochs, "
            f"{args.grad_accum_steps} accum steps/epoch)"
        )

        torch_path = output_dir / f"deep_relu_mnist_torch_{device.type}.pt"
        torch.save(_clone_state_dict_cpu(model), torch_path)
        print(f"\nSaved torch model (CPU tensors) to {torch_path}")


if __name__ == "__main__":
    main()
