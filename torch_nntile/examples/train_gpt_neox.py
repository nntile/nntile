#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_gpt_neox.py
# Tiny GPTNeoXCausal smoke train on device="nntile" (no tiling).

"""Short GPTNeoXCausal training smoke: synthetic tokens, few steps, print loss.

Example::

    python torch_nntile/examples/train_gpt_neox.py --steps 2 --seed 0 --ncpu 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "torch_nntile") not in sys.path:
    sys.path.insert(0, str(_REPO / "torch_nntile"))

import torch_nntile  # noqa: E402
from torch_nntile import _C  # noqa: E402
from torch_nntile.models.gpt_neox import (  # noqa: E402
    GPTNeoXCausal,
    GPTNeoXConfig,
)
from torch_nntile.training import AdamW, cross_entropy  # noqa: E402


def tiny_config() -> GPTNeoXConfig:
    return GPTNeoXConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=32,
        rotary_pct=0.25,
        tie_word_embeddings=False,
    )


def make_batch(
    vocab_size: int,
    *,
    batch_size: int,
    seq_len: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(
        0, vocab_size, (batch_size, seq_len + 1), dtype=torch.long, generator=g
    )
    return packed[:, :-1].clone(), packed[:, 1:].clone()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ncpu", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not _C.has_libnntile():
        raise SystemExit(
            "torch_nntile was built without libnntile. "
            "Set NNTILE_BUILD_DIR and reinstall."
        )
    torch.manual_seed(args.seed)
    cfg = tiny_config()
    inputs_cpu, labels_cpu = make_batch(
        cfg.vocab_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    model_cpu = GPTNeoXCausal(cfg).float().train()

    torch_nntile.init_context(
        ncpu=args.ncpu, ncuda=0, cpu_fallback=False
    )
    try:
        with torch.no_grad():
            inputs = inputs_cpu.to("nntile")
            labels = labels_cpu.to("nntile")
            model = model_cpu.to("nntile")
        del model_cpu, inputs_cpu, labels_cpu
        for p in model.parameters():
            p.requires_grad_(True)
        opt = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
        )
        opt.zero_grad(set_to_none=True)
        for step in range(args.steps):
            logits = model(inputs)
            loss = cross_entropy(logits, labels, reduction="mean")
            loss.backward()
            opt.step()
            step_loss = loss.detach()
            del loss, logits
            opt.zero_grad(set_to_none=True)
            torch_nntile.compile_graph()
            torch_nntile.run()
            torch_nntile.wait()
            value = float(step_loss.to("cpu").item())
            del step_loss
            print(
                f"[gpt_neox] step {step + 1}/{args.steps}  loss={value:.6f}"
            )
    finally:
        torch_nntile.shutdown_context()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
