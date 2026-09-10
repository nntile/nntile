# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/dit_hf_tiny_train_common.py
# Shared helpers for tiny Diffusers DiT smokes on cpu / nntile.

"""Tiny HuggingFace Diffusers DiT train loop (JSON config / checkpoint).

Mirrors :mod:`hf_tiny_train_common` for
``diffusers.DiTTransformer2DModel`` noise-prediction training on a small
``datasets`` image split (default: CIFAR-10).
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
import torchvision.transforms.functional as TF
from hf_tiny_train_common import (
    compare_checkpoints, configure_single_thread_host, load_json_object,
    save_checkpoint)
from nntile_iter_phases import (
    print_torch_iter_timings,
    run_nntile_train_iters,
)

LossFn = Callable[[nn.Module, dict[str, torch.Tensor]], torch.Tensor]
BatchBuilder = Callable[
    [Any, argparse.Namespace],
    dict[str, torch.Tensor],
]
ConfigFactory = Callable[..., Any]


def disable_dit_label_dropout(model: nn.Module) -> None:
    """Deterministic smokes: turn off CFG label dropout (``torch.rand``)."""
    for module in model.modules():
        if hasattr(module, "dropout_prob"):
            module.dropout_prob = 0.0


def load_dit_config_from_json(path: Path, config_cls: ConfigFactory) -> Any:
    fields = load_json_object(path)
    # Diffusers ConfigMixin: construct via model config class / kwargs.
    if hasattr(config_cls, "from_dict"):
        return config_cls.from_dict(fields)
    return config_cls(**fields)


def config_to_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    if isinstance(config, dict):
        return {k: v for k, v in config.items() if not str(k).startswith("_")}
    raise TypeError(f"config has no to_dict(): {type(config)!r}")


def make_cifar_diffusion_batch(
    *,
    batch_size: int,
    sample_size: int,
    in_channels: int,
    num_timesteps: int,
    num_classes: int,
    seed: int,
    dataset_name: str = "cifar10",
    dataset_split: str = "train[:64]",
) -> dict[str, torch.Tensor]:
    """Load a tiny ``datasets`` image split and build a noise-prediction batch.

    Images are resized to ``sample_size``, mapped to ``[-1, 1]``, and
    optionally expanded/truncated to ``in_channels``. Timesteps and class
    labels are drawn deterministically from ``seed``.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - env guard
        raise SystemExit(
            "dit HF smoke requires the HuggingFace datasets package"
        ) from exc

    g = torch.Generator().manual_seed(seed)
    ds = load_dataset(dataset_name, split=dataset_split)
    if len(ds) < batch_size:
        raise ValueError(
            f"dataset split too small: need {batch_size}, got {len(ds)}"
        )

    images: list[torch.Tensor] = []
    labels: list[int] = []
    for i in range(batch_size):
        row = ds[i]
        img = row["img"] if "img" in row else row["image"]
        label = int(row["label"])
        t = TF.to_tensor(img)  # CHW in [0, 1]
        t = TF.resize(
            t,
            [sample_size, sample_size],
            antialias=True,
        )
        if t.shape[0] == 1 and in_channels == 3:
            t = t.repeat(3, 1, 1)
        elif t.shape[0] >= in_channels:
            t = t[:in_channels]
        elif t.shape[0] < in_channels:
            pad = t.new_zeros(in_channels - t.shape[0], *t.shape[1:])
            t = torch.cat([t, pad], dim=0)
        t = t * 2.0 - 1.0
        images.append(t)
        labels.append(label % max(num_classes, 1))

    clean = torch.stack(images, dim=0).to(dtype=torch.float32)
    noise = torch.randn(
        clean.shape,
        dtype=torch.float32,
        generator=g,
    )
    # Linear alpha schedule (tiny smoke; not a full DDPM trainer).
    timesteps = torch.randint(
        0,
        num_timesteps,
        (batch_size,),
        dtype=torch.long,
        generator=g,
    )
    t_norm = timesteps.float() / float(max(num_timesteps - 1, 1))
    t_norm = t_norm.view(batch_size, 1, 1, 1)
    noisy = (1.0 - t_norm) * clean + t_norm * noise
    class_labels = torch.tensor(labels, dtype=torch.long)
    return {
        "noisy": noisy,
        "noise": noise,
        "timesteps": timesteps,
        "class_labels": class_labels,
    }


def make_synthetic_diffusion_batch(
    *,
    batch_size: int,
    sample_size: int,
    in_channels: int,
    num_timesteps: int,
    num_classes: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Deterministic noise-prediction batch without ``datasets`` I/O."""
    g = torch.Generator().manual_seed(seed)
    clean = torch.randn(
        batch_size,
        in_channels,
        sample_size,
        sample_size,
        dtype=torch.float32,
        generator=g,
    )
    noise = torch.randn(
        clean.shape,
        dtype=torch.float32,
        generator=g,
    )
    timesteps = torch.randint(
        0,
        num_timesteps,
        (batch_size,),
        dtype=torch.long,
        generator=g,
    )
    t_norm = timesteps.float() / float(max(num_timesteps - 1, 1))
    t_norm = t_norm.view(batch_size, 1, 1, 1)
    noisy = (1.0 - t_norm) * clean + t_norm * noise
    class_labels = torch.randint(
        0,
        max(num_classes, 1),
        (batch_size,),
        dtype=torch.long,
        generator=g,
    )
    return {
        "noisy": noisy,
        "noise": noise,
        "timesteps": timesteps,
        "class_labels": class_labels,
    }


def diffusion_mse_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Predict noise; MSE vs ground-truth noise."""
    out = model(
        batch["noisy"],
        timestep=batch["timesteps"],
        class_labels=batch["class_labels"],
        return_dict=True,
    )
    pred = out.sample if hasattr(out, "sample") else out[0]
    diff = pred - batch["noise"]
    return (diff * diff).mean()


def add_dit_train_compare_subparsers(
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
        help="Diffusers DiT JSON config path",
    )
    train.add_argument(
        "--output-dir",
        default="",
        help="Directory for checkpoint.pt (optional)",
    )
    train.add_argument("--steps", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--lr", type=float, default=1e-3)
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
        "--dataset",
        default="cifar10",
        help="HuggingFace datasets name (default: cifar10)",
    )
    train.add_argument(
        "--dataset-split",
        default="train[:64]",
        help="datasets split slice (default: train[:64])",
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
    model_cls: type,
) -> tuple[Any, nn.Module, int, int]:
    global_step = 0
    if args.checkpoint:
        ckpt = torch.load(
            Path(args.checkpoint),
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise ValueError(f"invalid checkpoint: {args.checkpoint}")
        config_dict = dict(ckpt["config"])
        model = model_cls.from_config(config_dict).float().train()
        model.load_state_dict(ckpt["model_state_dict"])
        disable_dit_label_dropout(model)
        global_step = int(ckpt.get("global_step", 0))
        seed = int(
            args.seed if args.seed is not None else ckpt.get("seed", 0)
        )
        print(
            f"Resumed from {args.checkpoint} "
            f"(step={global_step}, seed={seed})"
        )
        return model.config, model, seed, global_step

    if args.seed is None:
        raise SystemExit("--seed is required when training from scratch")
    fields = load_json_object(Path(args.config))
    torch.manual_seed(args.seed)
    model = model_cls(**fields).float().train()
    disable_dit_label_dropout(model)
    return model.config, model, int(args.seed), global_step


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


class _ConfigAdapter:
    """Wrap Diffusers config so ``save_checkpoint`` can call ``to_dict``."""

    def __init__(self, config: Any) -> None:
        self._config = config

    def to_dict(self) -> dict[str, Any]:
        return config_to_dict(self._config)


def run_tiny_dit_train(
    *,
    name: str,
    args: argparse.Namespace,
    config: Any,
    model: nn.Module,
    seed: int,
    global_step: int,
    build_batch: BatchBuilder,
    loss_fn: LossFn,
) -> int:
    configure_single_thread_host()
    print(
        f"=== {name} tiny DiT HF smoke  device={args.device}  "
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
                config=_ConfigAdapter(config),
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
                config=_ConfigAdapter(config),
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


def run_tiny_dit_hf_main(
    *,
    name: str,
    argv: list[str] | None,
    default_config: Path,
    model_cls: type,
    build_batch: BatchBuilder,
    loss_fn: LossFn,
    description: str = "",
) -> int:
    parser = argparse.ArgumentParser(
        description=description or f"Tiny Diffusers {name} smoke",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_dit_train_compare_subparsers(parser, default_config=default_config)
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
    return run_tiny_dit_train(
        name=name,
        args=args,
        config=config,
        model=model,
        seed=seed,
        global_step=global_step,
        build_batch=build_batch,
        loss_fn=loss_fn,
    )
