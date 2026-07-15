#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_t5.py
# Tiny T5ForConditionalGeneration smoke train on device="nntile" (no tiling).

"""Short T5 training smoke: encoder/decoder ids, few steps, print loss.

Example::

    python torch_nntile/examples/train_t5.py --steps 2 --seed 0 --ncpu 1
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
from torch_nntile.models.t5 import (  # noqa: E402
    T5Config,
    T5ForConditionalGeneration,
)
from torch_nntile.training import AdamW, cross_entropy  # noqa: E402


def tiny_config() -> T5Config:
    return T5Config(
        vocab_size=128,
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        tie_word_embeddings=False,
    )


def make_batch(
    vocab_size: int,
    *,
    batch_size: int,
    enc_len: int,
    dec_len: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encoder ids, decoder ids, and next-token labels for the decoder."""
    g = torch.Generator().manual_seed(seed)
    enc = torch.randint(
        0, vocab_size, (batch_size, enc_len), dtype=torch.long, generator=g
    )
    packed = torch.randint(
        0, vocab_size, (batch_size, dec_len + 1), dtype=torch.long, generator=g
    )
    dec = packed[:, :-1].clone()
    labels = packed[:, 1:].clone()
    return enc, dec, labels


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ncpu", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--enc-len", type=int, default=8)
    p.add_argument("--dec-len", type=int, default=8)
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
    enc_cpu, dec_cpu, labels_cpu = make_batch(
        cfg.vocab_size,
        batch_size=args.batch_size,
        enc_len=args.enc_len,
        dec_len=args.dec_len,
        seed=args.seed,
    )
    model_cpu = T5ForConditionalGeneration(cfg).float().train()

    torch_nntile.init_context(
        ncpu=args.ncpu, ncuda=0, cpu_fallback=False
    )
    try:
        with torch.no_grad():
            enc = enc_cpu.to("nntile")
            dec = dec_cpu.to("nntile")
            labels = labels_cpu.to("nntile")
            model = model_cpu.to("nntile")
            if cfg.tie_word_embeddings:
                model.lm_head.weight = model.model.shared.weight
        del model_cpu, enc_cpu, dec_cpu, labels_cpu
        for p in model.parameters():
            p.requires_grad_(True)
        opt = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
        )
        opt.zero_grad(set_to_none=True)
        for step in range(args.steps):
            logits = model(enc, dec)
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
            print(f"[t5] step {step + 1}/{args.steps}  loss={value:.6f}")
    finally:
        torch_nntile.shutdown_context()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
