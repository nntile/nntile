#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_roberta.py
# Tiny RobertaMlm smoke train on device="nntile" (no tiling).

"""Short RobertaMlm training smoke: random MLM labels, few steps, print loss.

Uses JSON config / checkpoint like ``train_gpt2_hf.py``::

    python torch_nntile/examples/train_roberta.py train \\
        --seed 0 --config roberta_tiny_config.json \\
        --output-dir /tmp/roberta --steps 2
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_nntile.models.roberta import RobertaConfig, RobertaMlm
from torch_nntile.training import cross_entropy

from nntile_tiny_train_common import run_tiny_nntile_main

IGNORE_INDEX = -100


def _default_config() -> Path:
    return Path(__file__).resolve().parent / "roberta_tiny_config.json"


def main(argv: list[str] | None = None) -> int:
    def build_batch(cfg, args):
        g = torch.Generator().manual_seed(args.seed)
        pad_token_id = int(cfg.pad_token_id)
        hi = max(cfg.vocab_size, 2)
        choices = [i for i in range(hi) if i != pad_token_id]
        idx = torch.randint(
            0,
            len(choices),
            (args.batch_size, args.seq_len),
            dtype=torch.long,
            generator=g,
        )
        input_ids = torch.tensor(choices, dtype=torch.long)[idx]
        labels = input_ids.clone()
        mask = (
            torch.rand(args.batch_size, args.seq_len, generator=g) < 0.25
        )
        if not bool(mask.any()):
            mask[0, 0] = True
        labels = labels.masked_fill(~mask, IGNORE_INDEX)
        input_ids = input_ids.clone()
        input_ids[mask] = 0
        return {"input_ids": input_ids, "labels": labels}

    def loss_fn(model, batch):
        logits = model(batch["input_ids"])
        return cross_entropy(
            logits,
            batch["labels"],
            reduction="mean",
            ignore_index=IGNORE_INDEX,
        )

    return run_tiny_nntile_main(
        name="roberta",
        argv=argv,
        default_config=_default_config(),
        config_cls=RobertaConfig,
        model_cls=RobertaMlm,
        build_batch=build_batch,
        loss_fn=loss_fn,
        description=__doc__,
    )


if __name__ == "__main__":
    raise SystemExit(main())
