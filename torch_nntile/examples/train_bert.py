#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_bert.py
# Tiny BertMlm smoke train on device="nntile" (no tiling).

"""Short BertMlm training smoke: random MLM labels, few steps, print loss.

Uses JSON config / checkpoint like ``train_gpt2_hf.py``::

    python torch_nntile/examples/train_bert.py train \\
        --seed 0 --config bert_tiny_config.json \\
        --output-dir /tmp/bert --steps 2
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_nntile.models.bert import BertConfig, BertMlm
from torch_nntile.training import cross_entropy

from nntile_tiny_train_common import run_tiny_nntile_main

IGNORE_INDEX = -100


def _default_config() -> Path:
    return Path(__file__).resolve().parent / "bert_tiny_config.json"


def main(argv: list[str] | None = None) -> int:
    def build_batch(cfg, args):
        g = torch.Generator().manual_seed(args.seed)
        input_ids = torch.randint(
            0,
            cfg.vocab_size,
            (args.batch_size, args.seq_len),
            dtype=torch.long,
            generator=g,
        )
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
        name="bert",
        argv=argv,
        default_config=_default_config(),
        config_cls=BertConfig,
        model_cls=BertMlm,
        build_batch=build_batch,
        loss_fn=loss_fn,
        description=__doc__,
    )


if __name__ == "__main__":
    raise SystemExit(main())
