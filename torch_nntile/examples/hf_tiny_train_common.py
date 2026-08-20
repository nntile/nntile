# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/hf_tiny_train_common.py
# Shared helpers for tiny HuggingFace stock-model smokes on nntile/cpu.

"""Shared tiny HF train loop with JSON config / checkpoint support.

Each model script points at a default ``*_hf_tiny_config.json``, then calls
:func:`run_tiny_hf_main` for ``train`` / ``compare`` (same checkpoint payload
shape as ``train_gpt2_hf.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

LossFn = Callable[[torch.nn.Module, dict[str, torch.Tensor]], torch.Tensor]


def configure_single_thread_host() -> None:
    """Pin host BLAS / PyTorch to one core for fair overhead comparisons.

    Call before any heavy compute. StarPU worker count is still controlled by
    ``init_context(ncpu=...)`` / ``--ncpu`` independently.
    """
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(key, "1")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # May already be set after the first parallel op.
        pass
BatchBuilder = Callable[
    [Any, argparse.Namespace],
    dict[str, torch.Tensor],
]
ConfigFactory = Callable[..., Any]
ModelFactory = Callable[[Any], torch.nn.Module]


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    return {k: v for k, v in raw.items() if not str(k).startswith("_")}


def load_hf_config_from_json(
    path: Path,
    config_cls: ConfigFactory,
    *,
    attn_implementation: str | None = None,
    use_cache: bool | None = None,
) -> Any:
    fields = load_json_object(path)
    config = config_cls(**fields)
    if attn_implementation is not None and hasattr(
        config, "_attn_implementation"
    ):
        config._attn_implementation = attn_implementation
    if use_cache is not None and hasattr(config, "use_cache"):
        config.use_cache = use_cache
    return config


def config_to_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    raise TypeError(f"config has no to_dict(): {type(config)!r}")


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    config: Any,
    seed: int,
    epoch: int,
    global_step: int,
    device_name: str,
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
        "config": config_to_dict(config),
        "seed": seed,
        "epoch": epoch,
        "global_step": global_step,
        "device": device_name,
        "optimizer_state_dict": optimizer_state,
    }
    torch.save(payload, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"invalid checkpoint format: {path}")
    return payload


def relative_frobenius(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> float:
    with torch.no_grad():
        diff = (a.float() - b.float()).norm().item()
        na = a.float().norm().item()
        nb = b.float().norm().item()
    return diff / max(na, nb, eps)


def compare_checkpoints(path_a: Path, path_b: Path) -> int:
    ckpt_a = load_checkpoint(path_a)
    ckpt_b = load_checkpoint(path_b)
    state_a = ckpt_a["model_state_dict"]
    state_b = ckpt_b["model_state_dict"]
    keys_a = set(state_a)
    keys_b = set(state_b)
    if keys_a != keys_b:
        only_a = sorted(keys_a - keys_b)
        only_b = sorted(keys_b - keys_a)
        print("WARNING: state_dict key mismatch")
        if only_a:
            print(f"  only in A ({len(only_a)}): {only_a[:8]}")
        if only_b:
            print(f"  only in B ({len(only_b)}): {only_b[:8]}")
    shared = sorted(keys_a & keys_b)
    print(f"Comparing {len(shared)} tensors")
    print(f"  A: {path_a}")
    print(f"  B: {path_b}")
    max_rel = 0.0
    worst = ""
    for name in shared:
        ta = state_a[name]
        tb = state_b[name]
        if ta.shape != tb.shape:
            print(
                f"  SKIP {name}: shape {tuple(ta.shape)} vs "
                f"{tuple(tb.shape)}"
            )
            continue
        rel = relative_frobenius(ta, tb)
        if rel > max_rel:
            max_rel = rel
            worst = name
        print(f"  {name}: relative_frobenius={rel:.6e}")
    print(f"max relative_frobenius={max_rel:.6e}  ({worst})")
    return 0


def add_train_compare_subparsers(
    parser: argparse.ArgumentParser,
    *,
    default_config: Path,
    devices: tuple[str, ...] = ("cpu", "nntile"),
) -> None:
    """Attach ``train`` / ``compare`` like ``train_gpt2_hf.py``."""
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
        help="HuggingFace JSON config path",
    )
    train.add_argument(
        "--output-dir",
        default="",
        help="Directory for checkpoint.pt (optional; skip save if empty)",
    )
    train.add_argument("--steps", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--seq-len", type=int, default=16)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument(
        "--ncpu",
        type=int,
        default=-1,
        help="StarPU CPU workers for --device nntile "
        "(default: -1 = STARPU_NCPU)",
    )
    train.add_argument(
        "--ncuda",
        type=int,
        default=-1,
        help="StarPU CUDA workers for --device nntile "
        "(default: -1 = STARPU_NCUDA)",
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
        help=(
            "Opt in to PyTorch CPU fallback for unregistered aten ops "
            "(implicit nntile<->CPU copies; off by default)"
        ),
    )

    compare = sub.add_parser(
        "compare",
        help="Print relative Frobenius norms between two checkpoints",
    )
    compare.add_argument("--checkpoint-a", required=True)
    compare.add_argument("--checkpoint-b", required=True)


def make_causal_batch(
    vocab_size: int,
    *,
    batch_size: int,
    seq_len: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len + 1),
        dtype=torch.long,
        generator=g,
    )
    return packed[:, :-1].clone(), packed[:, 1:].clone()


def make_mlm_batch(
    vocab_size: int,
    *,
    batch_size: int,
    seq_len: int,
    seed: int,
    mask_prob: float = 0.25,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len),
        dtype=torch.long,
        generator=g,
    )
    labels = input_ids.clone()
    mask = torch.rand(batch_size, seq_len, generator=g) < mask_prob
    if not bool(mask.any()):
        mask[0, 0] = True
    labels = labels.masked_fill(~mask, ignore_index)
    input_ids = input_ids.clone()
    input_ids[mask] = 0
    return input_ids, labels


def make_encoder_decoder_batch(
    vocab_size: int,
    *,
    batch_size: int,
    seq_len: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(encoder_ids, decoder_ids, labels)`` for T5-style CE."""
    g = torch.Generator().manual_seed(seed)
    enc = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len),
        dtype=torch.long,
        generator=g,
    )
    packed = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len + 1),
        dtype=torch.long,
        generator=g,
    )
    dec = packed[:, :-1].clone()
    labels = packed[:, 1:].clone()
    return enc, dec, labels


def causal_ce_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    logits = model(input_ids=batch["input_ids"]).logits
    vocab = logits.shape[-1]
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab),
        batch["labels"].reshape(-1),
    )


def mlm_ce_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    kwargs: dict[str, Any] = {"input_ids": batch["input_ids"]}
    if "attention_mask" in batch:
        kwargs["attention_mask"] = batch["attention_mask"]
    if "token_type_ids" in batch:
        kwargs["token_type_ids"] = batch["token_type_ids"]
    if "position_ids" in batch:
        # Avoid RoBERTa create_position_ids (aten::ne on pad id).
        kwargs["position_ids"] = batch["position_ids"]
    logits = model(**kwargs).logits
    vocab = logits.shape[-1]
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab),
        batch["labels"].reshape(-1),
        ignore_index=-100,
    )


def t5_ce_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    out = model(
        input_ids=batch["input_ids"],
        decoder_input_ids=batch["decoder_input_ids"],
        labels=batch["labels"],
    )
    if out.loss is not None:
        return out.loss
    logits = out.logits
    vocab = logits.shape[-1]
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab),
        batch["labels"].reshape(-1),
    )


def _load_train_state(
    args: argparse.Namespace,
    *,
    config_cls: ConfigFactory,
    model_cls: ModelFactory,
    attn_implementation: str | None,
    use_cache: bool | None,
) -> tuple[Any, torch.nn.Module, int, int]:
    """Return ``(config, cpu_model, seed, global_step)``."""
    global_step = 0
    if args.checkpoint:
        ckpt = load_checkpoint(Path(args.checkpoint))
        config = config_cls.from_dict(ckpt["config"])
        if attn_implementation is not None and hasattr(
            config, "_attn_implementation"
        ):
            config._attn_implementation = attn_implementation
        if use_cache is not None and hasattr(config, "use_cache"):
            config.use_cache = use_cache
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
    config = load_hf_config_from_json(
        Path(args.config),
        config_cls,
        attn_implementation=attn_implementation,
        use_cache=use_cache,
    )
    torch.manual_seed(args.seed)
    model = model_cls(config).float().train()
    return config, model, int(args.seed), global_step


def run_tiny_hf_train(
    *,
    name: str,
    args: argparse.Namespace,
    config: Any,
    model: torch.nn.Module,
    seed: int,
    global_step: int,
    build_batch: BatchBuilder,
    loss_fn: LossFn,
) -> int:
    """Run a few train steps; optionally save ``checkpoint.pt``."""
    configure_single_thread_host()
    print(
        f"=== {name} tiny HF smoke  device={args.device}  "
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
                config=config,
                seed=seed,
                epoch=0,
                global_step=global_step + args.steps,
                device_name="cpu",
            )
        return code

    import torch_nntile

    ncuda = int(getattr(args, "ncuda", -1))
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
                config=config,
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


def _train_loop(
    *,
    name: str,
    model: torch.nn.Module,
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
    t0 = time.perf_counter()
    for step in range(steps):
        loss = loss_fn(model, batch)
        loss.backward()
        opt.step()
        if torch_nntile is not None:
            # Drop autograd before compile so activation tiles unmark.
            step_loss = loss.detach()
            del loss
            opt.zero_grad(set_to_none=True)
            torch_nntile.compile_graph()
            torch_nntile.run()
            with torch.no_grad():
                loss_val = float(step_loss.to("cpu").item())
            del step_loss
        else:
            loss_val = float(loss.detach())
            del loss
            opt.zero_grad(set_to_none=True)
        print(f"[{name}] step {step + 1}/{steps}  loss={loss_val:.6f}")
    print(f"[{name}] wall={time.perf_counter() - t0:.3f}s  OK")
    return 0


def run_tiny_hf_main(
    *,
    name: str,
    argv: list[str] | None,
    default_config: Path,
    config_cls: ConfigFactory,
    model_cls: ModelFactory,
    build_batch: BatchBuilder,
    loss_fn: LossFn,
    attn_implementation: str | None = None,
    use_cache: bool | None = None,
    description: str = "",
) -> int:
    """CLI entry: ``train`` / ``compare`` with JSON config or checkpoint."""
    parser = argparse.ArgumentParser(
        description=description or f"Tiny HF {name} smoke",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_train_compare_subparsers(parser, default_config=default_config)
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
        attn_implementation=attn_implementation,
        use_cache=use_cache,
    )
    args.seed = seed
    return run_tiny_hf_train(
        name=name,
        args=args,
        config=config,
        model=model,
        seed=seed,
        global_step=global_step,
        build_batch=build_batch,
        loss_fn=loss_fn,
    )


def extract_missing_op(exc: BaseException) -> str | None:
    """Best-effort parse of PrivateUse1 'not implemented' messages."""
    msg = str(exc)
    markers = (
        "is not currently implemented for the PrivateUse1",
        "not implemented for 'PrivateUse1'",
        "Could not run '",
        "is disabled under NNTILE_TORCH_NATIVE_OPS",
    )
    for m in markers:
        if m in msg:
            return msg
    return None
