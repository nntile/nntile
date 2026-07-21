# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/dit_hf_tiny_train_common.py
# Shared helpers for tiny Diffusers DiT smokes on cpu / cuda / nntile.

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
    compare_checkpoints,
    configure_single_thread_host,
    configure_tf32,
    config_to_dict,
    load_json_object,
    save_checkpoint,
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
    devices: tuple[str, ...] = ("cpu", "cuda", "nntile"),
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
    train.add_argument(
        "--disable-tf32",
        action="store_true",
        help="Disable TF32 for full FP32 matmul (cuda / nntile CUDA)",
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
    sync_cuda: bool = False,
) -> int:
    torch_nntile = None
    if nntile:
        import torch_nntile as _tn

        torch_nntile = _tn

    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )
    t0 = time.perf_counter()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        if torch_nntile is not None:
            torch_nntile.compile_graph()
            torch_nntile.run()
        loss = loss_fn(model, batch)
        loss.backward()
        opt.step()
        if torch_nntile is not None:
            loss_val = float(loss.detach().cpu())
        else:
            loss_val = float(loss.detach().item())
        print(f"[{name}] step {step + 1}/{steps}  loss={loss_val:.6f}")
        # Reclaim StarPU temps: drop the step loss before next compile.
        del loss
    if sync_cuda:
        torch.cuda.synchronize()
    print(f"[{name}] wall={time.perf_counter() - t0:.3f}s  OK")
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
    if (
        getattr(args, "disable_tf32", False)
        or args.device == "cuda"
        or int(getattr(args, "ncuda", 0)) > 0
    ):
        configure_tf32(disable_tf32=True)
    print(
        f"=== {name} tiny DiT HF smoke  device={args.device}  "
        f"config_seed={seed} ==="
    )
    batch_cpu = build_batch(config, args)

    if args.device in ("cpu", "cuda"):
        if args.device == "cuda" and not torch.cuda.is_available():
            print("FAIL: CUDA is not available")
            return 1
        device = torch.device(args.device)
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch_cpu.items()}
            model = model.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        code = _train_loop(
            name=name,
            model=model,
            batch=batch,
            loss_fn=loss_fn,
            steps=args.steps,
            lr=args.lr,
            nntile=False,
            sync_cuda=device.type == "cuda",
        )
        if code == 0 and args.output_dir:
            save_checkpoint(
                Path(args.output_dir) / "checkpoint.pt",
                model=model,
                config=_ConfigAdapter(config),
                seed=seed,
                epoch=0,
                global_step=global_step + args.steps,
                device_name=args.device,
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
