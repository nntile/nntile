#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_t5.py
# Tiny T5ForConditionalGeneration smoke train on device="nntile" (no tiling).

"""Short T5 training smoke: encoder/decoder ids, few steps, print loss.

Uses JSON config / checkpoint like ``train_gpt2_hf.py``::

    python torch_nntile/examples/train_t5.py train \\
        --seed 0 --config t5_tiny_config.json \\
        --output-dir /tmp/t5 --steps 2
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_nntile.models.t5 import T5Config, T5ForConditionalGeneration
from torch_nntile.training import cross_entropy

from nntile_tiny_train_common import run_tiny_nntile_main


def _default_config() -> Path:
    return Path(__file__).resolve().parent / "t5_tiny_config.json"


def main(argv: list[str] | None = None) -> int:
    def build_batch(cfg, args):
        enc_len = (
            args.enc_len if args.enc_len is not None else args.seq_len
        )
        dec_len = (
            args.dec_len if args.dec_len is not None else args.seq_len
        )
        g = torch.Generator().manual_seed(args.seed)
        enc = torch.randint(
            0,
            cfg.vocab_size,
            (args.batch_size, enc_len),
            dtype=torch.long,
            generator=g,
        )
        packed = torch.randint(
            0,
            cfg.vocab_size,
            (args.batch_size, dec_len + 1),
            dtype=torch.long,
            generator=g,
        )
        return {
            "input_ids": enc,
            "decoder_input_ids": packed[:, :-1].clone(),
            "labels": packed[:, 1:].clone(),
        }

    def loss_fn(model, batch):
        logits = model(batch["input_ids"], batch["decoder_input_ids"])
        return cross_entropy(
            logits, batch["labels"], reduction="mean"
        )

    return run_tiny_nntile_main(
        name="t5",
        argv=argv,
        default_config=_default_config(),
        config_cls=T5Config,
        model_cls=T5ForConditionalGeneration,
        build_batch=build_batch,
        loss_fn=loss_fn,
        description=__doc__,
    )


if __name__ == "__main__":
    raise SystemExit(main())
