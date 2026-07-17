#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_roberta.py
# Tiny RobertaMlm smoke train on device="nntile" (no tiling).

"""Short RobertaMlm training smoke: random MLM labels, few steps, print loss.

Example::

    python torch_nntile/examples/train_roberta.py --steps 2 --seed 0 --ncpu 1
"""

from __future__ import annotations

import argparse

import torch

import torch_nntile
from torch_nntile.models.roberta import RobertaConfig, RobertaMlm
from torch_nntile.training import AdamW, cross_entropy

IGNORE_INDEX = -100


def tiny_config() -> RobertaConfig:
    return RobertaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=34,
        type_vocab_size=1,
        # nntile Embedding requires padding_idx == -1.
        pad_token_id=-1,
    )


def make_mlm_batch(
    vocab_size: int,
    *,
    batch_size: int,
    seq_len: int,
    seed: int,
    pad_token_id: int,
    mask_prob: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random input ids + MLM labels (``IGNORE_INDEX`` on unmasked tokens)."""
    g = torch.Generator().manual_seed(seed)
    # Avoid pad_token_id in content so position ids stay dense.
    hi = max(vocab_size, 2)
    choices = [i for i in range(hi) if i != pad_token_id]
    idx = torch.randint(
        0, len(choices), (batch_size, seq_len), dtype=torch.long, generator=g
    )
    input_ids = torch.tensor(choices, dtype=torch.long)[idx]
    labels = input_ids.clone()
    mask = torch.rand(batch_size, seq_len, generator=g) < mask_prob
    if not bool(mask.any()):
        mask[0, 0] = True
    labels = labels.masked_fill(~mask, IGNORE_INDEX)
    input_ids = input_ids.clone()
    input_ids[mask] = 0
    return input_ids, labels


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
    torch.manual_seed(args.seed)
    cfg = tiny_config()
    inputs_cpu, labels_cpu = make_mlm_batch(
        cfg.vocab_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed,
        pad_token_id=cfg.pad_token_id,
    )
    model_cpu = RobertaMlm(cfg).float().train()

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
            loss = cross_entropy(
                logits,
                labels,
                reduction="mean",
                ignore_index=IGNORE_INDEX,
            )
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
            print(f"[roberta] step {step + 1}/{args.steps}  loss={value:.6f}")
    finally:
        torch_nntile.shutdown_context()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
