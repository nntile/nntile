# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/hf_tiny_train_common.py
# Shared helpers for tiny HuggingFace stock-model smokes on nntile/cpu.

"""Shared tiny HF train loop for discovering missing torch-native ops.

Each model script builds a tiny config + model, then calls
:func:`run_tiny_hf_train` for a short synthetic step on ``cpu`` or
``nntile``.
"""

from __future__ import annotations

import argparse
import time
import traceback
from collections.abc import Callable
from typing import Any

import torch


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=("cpu", "nntile"),
        default="nntile",
        help="Training device (cuda not used on this smoke path)",
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ncpu", type=int, default=2)
    parser.add_argument(
        "--cpu-fallback",
        action="store_true",
        help="Allow unregistered aten ops to fall back to CPU "
        "(hides missing PrivateUse1 coverage)",
    )


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


LossFn = Callable[[torch.nn.Module, dict[str, torch.Tensor]], torch.Tensor]


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
    # Prefer HF fused loss when present; else CE on logits.
    if out.loss is not None:
        return out.loss
    logits = out.logits
    vocab = logits.shape[-1]
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab),
        batch["labels"].reshape(-1),
    )


def run_tiny_hf_train(
    *,
    name: str,
    args: argparse.Namespace,
    build_model: Callable[[], torch.nn.Module],
    build_batch: Callable[[], dict[str, torch.Tensor]],
    loss_fn: LossFn,
) -> int:
    """Run a few train steps; return 0 on success, 1 on failure."""
    torch.manual_seed(args.seed)
    print(f"=== {name} tiny HF smoke  device={args.device} ===")

    cpu_model = build_model().float().train()
    batch_cpu = build_batch()

    if args.device == "cpu":
        model = cpu_model
        batch = {k: v.clone() for k, v in batch_cpu.items()}
        return _train_loop(
            name=name,
            model=model,
            batch=batch,
            loss_fn=loss_fn,
            steps=args.steps,
            lr=args.lr,
            sync_fn=lambda: None,
            after_step=None,
        )

    import torch_nntile

    torch_nntile.init_context(
        ncpu=args.ncpu,
        ncuda=0,
        verbose=0,
        cpu_fallback=bool(args.cpu_fallback),
    )
    torch_nntile.restrict_cpu()
    try:
        with torch.no_grad():
            batch = {k: v.to("nntile") for k, v in batch_cpu.items()}
            model = cpu_model.to("nntile")
        torch_nntile.compile_graph()
        torch_nntile.run()
        del cpu_model
        for p in model.parameters():
            p.requires_grad_(True)

        def after_step() -> None:
            # Host sync + reclaim: loss .cpu() already flushes; keep
            # INVALIDATEs in the same phase as zero_grad when possible.
            pass

        return _train_loop(
            name=name,
            model=model,
            batch=batch,
            loss_fn=loss_fn,
            steps=args.steps,
            lr=args.lr,
            sync_fn=lambda: None,
            after_step=after_step,
            nntile=True,
        )
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
    sync_fn: Callable[[], None],
    after_step: Callable[[], None] | None,
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
            # Materialize loss on host (also flushes pending graph).
            loss_val = float(loss.detach().cpu())
        else:
            loss_val = float(loss.detach())
        sync_fn()
        if after_step is not None:
            after_step()
        print(f"[{name}] step {step + 1}/{steps}  loss={loss_val:.6f}")
    print(f"[{name}] wall={time.perf_counter() - t0:.3f}s  OK")
    return 0


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
