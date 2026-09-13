# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/nntile_tiny_train_common.py
# Shared helpers for tiny torch_nntile model smokes (JSON config / checkpoint).

"""Shared nntile-native tiny train loop with JSON config / checkpoint.

Mirrors the ``train`` / ``compare`` UX of ``train_gpt2_hf.py`` for the
hand-written ``torch_nntile.nn.model.*`` stacks (Llama, BERT, …).

Pass ``--ddp`` (and ``--ncpu`` / ``--ncuda``) to shard the named batch
axis. ``--batch-size`` must be at least the replica count. HuggingFace
``train_*_hf.py`` scripts cannot use DDP (stock aten stays untiled).
"""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from hf_tiny_train_common import (
    compare_checkpoints, load_checkpoint, load_json_object)
from nntile_iter_phases import run_nntile_train_iters

import torch_nntile
from torch_nntile.training import AdamW

BatchBuilder = Callable[
    [Any, argparse.Namespace],
    dict[str, torch.Tensor],
]
ConfigFactory = Callable[..., Any]
ModelFactory = Callable[[Any], torch.nn.Module]
LossFn = Callable[
    [torch.nn.Module, dict[str, torch.Tensor]],
    torch.Tensor,
]


def load_dataclass_config(
    path: Path,
    config_cls: ConfigFactory,
) -> Any:
    fields = load_json_object(path)
    if dataclasses.is_dataclass(config_cls):
        allowed = {f.name for f in dataclasses.fields(config_cls)}
        fields = {k: v for k, v in fields.items() if k in allowed}
    return config_cls(**fields)


def dataclass_config_to_dict(config: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    raise TypeError(f"unsupported config type: {type(config)!r}")


def save_nntile_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    config: Any,
    seed: int,
    epoch: int,
    global_step: int,
    device_name: str = "nntile",
    optimizer_state: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
    payload = {
        "model_state_dict": state,
        "config": dataclass_config_to_dict(config),
        "seed": seed,
        "epoch": epoch,
        "global_step": global_step,
        "device": device_name,
        "optimizer_state_dict": optimizer_state,
    }
    torch.save(payload, path)
    print(f"Saved checkpoint to {path}")


def add_nntile_train_compare_subparsers(
    parser: argparse.ArgumentParser,
    *,
    default_config: Path,
) -> None:
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser(
        "train",
        help="Train from JSON config or resume a checkpoint",
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
        help="JSON config path for the nntile model",
    )
    train.add_argument(
        "--output-dir",
        default="",
        help="Directory for checkpoint.pt (optional; skip save if empty)",
    )
    train.add_argument("--steps", type=int, default=2)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--seq-len", type=int, default=8)
    train.add_argument(
        "--enc-len",
        type=int,
        default=None,
        help="Encoder length for T5-style models (default: --seq-len)",
    )
    train.add_argument(
        "--dec-len",
        type=int,
        default=None,
        help="Decoder length for T5-style models (default: --seq-len)",
    )
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--ncpu", type=int, default=1)
    train.add_argument(
        "--ncuda",
        type=int,
        default=0,
        help="StarPU CUDA workers (replica count when > 0)",
    )
    train.add_argument(
        "--ddp",
        action="store_true",
        help="Shard the named batch axis across StarPU workers",
    )
    train.add_argument(
        "--ddp-axis",
        default="batch",
        help="Named axis for --ddp (default: batch)",
    )
    train.add_argument(
        "--print-axis-groups",
        action="store_true",
        help="Print pending axis groups after ddp() (before compile)",
    )

    compare = sub.add_parser(
        "compare",
        help="Print relative Frobenius norms between two checkpoints",
    )
    compare.add_argument("--checkpoint-a", required=True)
    compare.add_argument("--checkpoint-b", required=True)


def name_ddp_batch_axis(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    axis: str = "batch",
) -> None:
    """Name dim 0 of batched compute inputs (tokens, labels, ...).

    RoPE ``sin``/``cos`` are ``[seq, head_dim // 2]`` and must not join
    the DDP batch group: the kernel applies them across heads / batch.
    Cached ``position_ids`` are unused in compute.
    """
    del model
    for tensor in batch.values():
        if tensor.ndim >= 1:
            torch_nntile.set_axis_group_name(tensor, {0: axis})


def _load_train_state(
    args: argparse.Namespace,
    *,
    config_cls: ConfigFactory,
    model_cls: ModelFactory,
) -> tuple[Any, torch.nn.Module, int, int]:
    global_step = 0
    if args.checkpoint:
        ckpt = load_checkpoint(Path(args.checkpoint))
        fields = dict(ckpt["config"])
        if dataclasses.is_dataclass(config_cls):
            allowed = {f.name for f in dataclasses.fields(config_cls)}
            fields = {k: v for k, v in fields.items() if k in allowed}
        config = config_cls(**fields)
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
    config = load_dataclass_config(Path(args.config), config_cls)
    torch.manual_seed(args.seed)
    model = model_cls(config).float().train()
    return config, model, int(args.seed), global_step


def run_tiny_nntile_train(
    *,
    name: str,
    args: argparse.Namespace,
    config: Any,
    model_cpu: torch.nn.Module,
    seed: int,
    global_step: int,
    build_batch: BatchBuilder,
    loss_fn: LossFn,
) -> int:
    print(f"=== {name} tiny nntile smoke  seed={seed} ===")
    batch_cpu = build_batch(config, args)
    ddp = bool(getattr(args, "ddp", False))
    ddp_axis = str(getattr(args, "ddp_axis", "batch"))
    print_groups = bool(getattr(args, "print_axis_groups", False))
    ncuda = int(getattr(args, "ncuda", 0))

    replicas = ncuda if ncuda > 0 else int(args.ncpu)
    if ddp and replicas > 1 and int(args.batch_size) < replicas:
        raise SystemExit(
            f"--ddp needs --batch-size >= replica count "
            f"({replicas} = ncuda if ncuda>0 else ncpu)"
        )

    torch_nntile.init_context(
        ncpu=args.ncpu, ncuda=ncuda, cpu_fallback=False
    )
    try:
        with torch.no_grad():
            batch = {k: v.to("nntile") for k, v in batch_cpu.items()}
            model = model_cpu.to("nntile")
        del model_cpu, batch_cpu

        printed_groups = 0

        def apply_ddp() -> None:
            nonlocal printed_groups
            if not ddp:
                return
            name_ddp_batch_axis(model, batch, ddp_axis)
            torch_nntile.ddp(ddp_axis)
            if print_groups and printed_groups < 2:
                torch_nntile.print_axis_groups()
                printed_groups += 1

        apply_ddp()
        torch_nntile.compile_graph()
        torch_nntile.run()
        torch_nntile.wait()
        for p in model.parameters():
            p.requires_grad_(True)
        opt = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
        )
        opt.zero_grad(set_to_none=True)
        code = run_nntile_train_iters(
            name=name,
            model=model,
            batch=batch,
            loss_fn=loss_fn,
            steps=args.steps,
            opt=opt,
            torch_nntile=torch_nntile,
            before_compile=apply_ddp if ddp else None,
        )
        if code == 0 and args.output_dir:
            save_nntile_checkpoint(
                Path(args.output_dir) / "checkpoint.pt",
                model=model,
                config=config,
                seed=seed,
                epoch=0,
                global_step=global_step + args.steps,
            )
        return code
    finally:
        torch_nntile.shutdown_context()


def run_tiny_nntile_main(
    *,
    name: str,
    argv: list[str] | None,
    default_config: Path,
    config_cls: ConfigFactory,
    model_cls: ModelFactory,
    build_batch: BatchBuilder,
    loss_fn: LossFn,
    description: str = "",
) -> int:
    parser = argparse.ArgumentParser(
        description=description or f"Tiny nntile {name} smoke",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_nntile_train_compare_subparsers(
        parser, default_config=default_config
    )
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
        config_cls=config_cls,
        model_cls=model_cls,
    )
    args.seed = seed
    return run_tiny_nntile_train(
        name=name,
        args=args,
        config=config,
        model_cpu=model,
        seed=seed,
        global_step=global_step,
        build_batch=build_batch,
        loss_fn=loss_fn,
    )


# Re-export for callers that only import this module.
__all__ = [
    "compare_checkpoints",
    "load_checkpoint",
    "load_dataclass_config",
    "name_ddp_batch_axis",
    "run_tiny_nntile_main",
    "save_nntile_checkpoint",
]
