#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_llama.py
# Tiny LlamaCausal smoke train on device="nntile" (no tiling).

"""Short LlamaCausal training smoke: synthetic tokens, few steps, print loss.

Uses JSON config / checkpoint like ``train_gpt2_hf.py``::

    python torch_nntile/examples/train_llama.py train \\
        --seed 0 --config llama_tiny_config.json \\
        --output-dir /tmp/llama --steps 2
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_nntile.models.llama import LlamaCausal, LlamaConfig
from torch_nntile.training import cross_entropy

from nntile_tiny_train_common import run_tiny_nntile_main


def _default_config() -> Path:
    return Path(__file__).resolve().parent / "llama_tiny_config.json"


def main(argv: list[str] | None = None) -> int:
    def build_batch(cfg, args):
        g = torch.Generator().manual_seed(args.seed)
        packed = torch.randint(
            0,
            cfg.vocab_size,
            (args.batch_size, args.seq_len + 1),
            dtype=torch.long,
            generator=g,
        )
        return {
            "input_ids": packed[:, :-1].clone(),
            "labels": packed[:, 1:].clone(),
        }

    def loss_fn(model, batch):
        logits = model(batch["input_ids"])
        return cross_entropy(
            logits, batch["labels"], reduction="mean"
        )

    return run_tiny_nntile_main(
        name="llama",
        argv=argv,
        default_config=_default_config(),
        config_cls=LlamaConfig,
        model_cls=LlamaCausal,
        build_batch=build_batch,
        loss_fn=loss_fn,
        description=__doc__,
    )


if __name__ == "__main__":
    raise SystemExit(main())
