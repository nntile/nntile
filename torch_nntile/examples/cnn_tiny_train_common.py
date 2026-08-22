# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/cnn_tiny_train_common.py
# Shared helpers for tiny CNN smokes on cpu / nntile.

"""Shared tiny CNN train loop (JSON config / checkpoint).

Mirrors :mod:`hf_tiny_train_common` for vision models that exercise the
torch-native StarPU CNN ops (``conv2d``, pooling, ``batch_norm``).
"""

from __future__ import annotations

import argparse
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from hf_tiny_train_common import (
    compare_checkpoints, configure_single_thread_host, load_json_object,
    save_checkpoint)
from nntile_iter_phases import (
    print_torch_iter_timings,
    run_nntile_train_iters,
)

LossFn = Callable[[nn.Module, dict[str, torch.Tensor]], torch.Tensor]
BatchBuilder = Callable[
    [dict[str, Any], argparse.Namespace],
    dict[str, torch.Tensor],
]
ModelFactory = Callable[[dict[str, Any]], nn.Module]


def make_image_batch(
    *,
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    num_classes: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Deterministic synthetic images + labels (no external dataset)."""
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(
        batch_size,
        channels,
        height,
        width,
        dtype=torch.float32,
        generator=g,
    )
    labels = torch.randint(
        0,
        num_classes,
        (batch_size,),
        dtype=torch.long,
        generator=g,
    )
    return {"images": images, "labels": labels}


def make_segmentation_batch(
    *,
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    num_classes: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Deterministic synthetic images + per-pixel class labels."""
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(
        batch_size,
        channels,
        height,
        width,
        dtype=torch.float32,
        generator=g,
    )
    labels = torch.randint(
        0,
        num_classes,
        (batch_size, height, width),
        dtype=torch.long,
        generator=g,
    )
    return {"images": images, "labels": labels}


def classification_ce_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    logits = model(batch["images"])
    return nn.functional.cross_entropy(logits, batch["labels"])


def segmentation_ce_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Pixel-wise CE for NCHW logits vs NHW long labels.

    Flattens spatial dims so loss uses ``nll_loss`` (1D) rather than
    ``nll_loss2d``, which is not registered on PrivateUse1 yet.
    """
    logits = model(batch["images"])
    # logits: NCHW -> (N*H*W, C); labels: NHW -> (N*H*W,)
    n, c, h, w = logits.shape
    logits_flat = (
        logits.permute(0, 2, 3, 1).contiguous().reshape(n * h * w, c)
    )
    labels_flat = batch["labels"].reshape(n * h * w)
    return nn.functional.cross_entropy(logits_flat, labels_flat)


def add_cnn_train_compare_subparsers(
    parser: argparse.ArgumentParser,
    *,
    default_config: Path,
    devices: tuple[str, ...] = ("cpu", "nntile"),
) -> None:
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser(
        "train",
        help="Train from JSON config or resume a checkpoint",
    )
    train.add_argument(
        "--device",
        choices=devices,
        default="nntile",
        help="Training device",
    )
    train.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (required when training from scratch)",
    )
    train.add_argument(
        "--checkpoint",
        default="",
        help="Resume weights from this checkpoint",
    )
    train.add_argument(
        "--config",
        default=str(default_config),
        help="JSON model config path",
    )
    train.add_argument(
        "--output-dir",
        default="",
        help="Directory for checkpoint.pt (optional)",
    )
    train.add_argument("--steps", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--lr", type=float, default=1e-2)
    train.add_argument(
        "--ncpu",
        type=int,
        default=1,
        help="StarPU CPU workers for --device nntile (default: 1)",
    )
    train.add_argument(
        "--ncuda",
        type=int,
        default=0,
        help="StarPU CUDA workers for --device nntile (default: 0)",
    )
    train.add_argument(
        "--restrict-cpu",
        action="store_true",
        help="restrict_cpu() after init (nntile only)",
    )
    train.add_argument(
        "--restrict-cuda",
        action="store_true",
        help="restrict_cuda() after init (nntile only)",
    )
    train.add_argument(
        "--cpu-fallback",
        action="store_true",
        help="Allow unregistered aten ops to fall back to CPU",
    )

    compare = sub.add_parser(
        "compare",
        help="Print relative Frobenius norms between two checkpoints",
    )
    compare.add_argument("--checkpoint-a", required=True)
    compare.add_argument("--checkpoint-b", required=True)


def _load_train_state(
    args: argparse.Namespace,
    *,
    model_cls: ModelFactory,
) -> tuple[dict[str, Any], nn.Module, int, int]:
    global_step = 0
    if args.checkpoint:
        ckpt = torch.load(
            Path(args.checkpoint),
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise ValueError(f"invalid checkpoint: {args.checkpoint}")
        config = dict(ckpt["config"])
        model = model_cls(config).float().train()
        model.load_state_dict(ckpt["model_state_dict"])
        global_step = int(ckpt.get("global_step", 0))
        seed = int(
            args.seed if args.seed is not None else ckpt.get("seed", 0)
        )
        print(
            f"Resumed from {args.checkpoint} "
            f"(step={global_step}, seed={seed})"
        )
        return config, model, seed, global_step

    if args.seed is None:
        raise SystemExit("--seed is required when training from scratch")
    config = load_json_object(Path(args.config))
    torch.manual_seed(args.seed)
    model = model_cls(config).float().train()
    return config, model, int(args.seed), global_step


def _train_loop(
    *,
    name: str,
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    loss_fn: LossFn,
    steps: int,
    lr: float,
    nntile: bool = False,
) -> int:
    torch_nntile = None
    if nntile:
        import torch_nntile as _tn

        torch_nntile = _tn

    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )
    opt.zero_grad(set_to_none=True)
    last_loss: torch.Tensor | None = None
    if torch_nntile is not None:
        return run_nntile_train_iters(
            name=name,
            model=model,
            batch=batch,
            loss_fn=loss_fn,
            steps=steps,
            opt=opt,
            torch_nntile=torch_nntile,
        )

    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(steps):
        t_iter0 = time.perf_counter()
        loss = loss_fn(model, batch)
        loss.backward()
        opt.step()
        step_loss = loss.detach()
        del loss
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print_torch_iter_timings(
            step + 1,
            steps,
            time.perf_counter() - t_iter0,
        )
        if step == steps - 1:
            last_loss = step_loss
        else:
            del step_loss
    wall_s = time.perf_counter() - t0
    if last_loss is None:
        raise RuntimeError(f"{name}: no steps ran")
    loss_val = float(last_loss.item())
    del last_loss
    print(f"[{name}] final loss={loss_val:.6f}")
    print(f"[{name}] wall={wall_s:.3f}s  OK")
    return 0


def run_tiny_cnn_train(
    *,
    name: str,
    args: argparse.Namespace,
    config: dict[str, Any],
    model: nn.Module,
    seed: int,
    global_step: int,
    build_batch: BatchBuilder,
    loss_fn: LossFn,
) -> int:
    configure_single_thread_host()
    print(
        f"=== {name} tiny CNN smoke  device={args.device}  "
        f"config_seed={seed} ==="
    )
    batch_cpu = build_batch(config, args)

    if args.device == "cpu":
        code = _train_loop(
            name=name,
            model=model,
            batch={k: v.clone() for k, v in batch_cpu.items()},
            loss_fn=loss_fn,
            steps=args.steps,
            lr=args.lr,
            nntile=False,
        )
        if code == 0 and args.output_dir:
            save_checkpoint(
                Path(args.output_dir) / "checkpoint.pt",
                model=model,
                config=_ConfigDict(config),
                seed=seed,
                epoch=0,
                global_step=global_step + args.steps,
                device_name="cpu",
            )
        return code

    import torch_nntile

    ncuda = int(getattr(args, "ncuda", 0))
    torch_nntile.init_context(
        ncpu=args.ncpu,
        ncuda=ncuda,
        verbose=0,
        cpu_fallback=bool(args.cpu_fallback),
    )
    if getattr(args, "restrict_cuda", False):
        torch_nntile.restrict_cuda()
    elif getattr(args, "restrict_cpu", False) or ncuda == 0:
        torch_nntile.restrict_cpu()
    try:
        with torch.no_grad():
            batch = {k: v.to("nntile") for k, v in batch_cpu.items()}
            model = model.to("nntile")
        torch_nntile.compile_graph()
        torch_nntile.run()
        torch_nntile.wait()
        for p in model.parameters():
            p.requires_grad_(True)

        code = _train_loop(
            name=name,
            model=model,
            batch=batch,
            loss_fn=loss_fn,
            steps=args.steps,
            lr=args.lr,
            nntile=True,
        )
        if code == 0 and args.output_dir:
            save_checkpoint(
                Path(args.output_dir) / "checkpoint.pt",
                model=model,
                config=_ConfigDict(config),
                seed=seed,
                epoch=0,
                global_step=global_step + args.steps,
                device_name="nntile",
            )
        return code
    except Exception as exc:  # noqa: BLE001 — discovery harness
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    finally:
        try:
            torch_nntile.shutdown_context()
        except Exception:  # noqa: BLE001
            pass


class _ConfigDict:
    """Thin wrapper so ``save_checkpoint`` can call ``to_dict()``."""

    def __init__(self, fields: dict[str, Any]) -> None:
        self._fields = dict(fields)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._fields)


def run_tiny_cnn_main(
    *,
    name: str,
    argv: list[str] | None,
    default_config: Path,
    model_cls: ModelFactory,
    build_batch: BatchBuilder,
    loss_fn: LossFn,
    description: str = "",
) -> int:
    parser = argparse.ArgumentParser(
        description=description or f"Tiny CNN {name} smoke",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_cnn_train_compare_subparsers(parser, default_config=default_config)
    args = parser.parse_args(argv)
    if args.command == "compare":
        return compare_checkpoints(
            Path(args.checkpoint_a),
            Path(args.checkpoint_b),
        )
    if args.command != "train":
        raise SystemExit(f"unknown command: {args.command}")
    if not args.checkpoint and args.seed is None:
        raise SystemExit("--seed is required when training from scratch")

    config, model, seed, global_step = _load_train_state(
        args,
        model_cls=model_cls,
    )
    args.seed = seed
    return run_tiny_cnn_train(
        name=name,
        args=args,
        config=config,
        model=model,
        seed=seed,
        global_step=global_step,
        build_batch=build_batch,
        loss_fn=loss_fn,
    )
