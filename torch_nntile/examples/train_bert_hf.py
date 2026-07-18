#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_bert_hf.py
# Tiny stock HuggingFace BertForMaskedLM smoke on cpu/nntile.

"""Tiny HF BERT MLM smoke (synthetic tokens).

Uses JSON config / checkpoint like ``train_gpt2_hf.py``::

    python torch_nntile/examples/train_bert_hf.py train \\
        --device nntile --seed 0 --config bert_hf_tiny_config.json \\
        --output-dir /tmp/bert_hf --steps 1
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import BertConfig, BertForMaskedLM

from hf_tiny_train_common import (
    make_mlm_batch,
    mlm_ce_loss,
    run_tiny_hf_main,
)


def _default_config() -> Path:
    return Path(__file__).resolve().parent / "bert_hf_tiny_config.json"


def main(argv: list[str] | None = None) -> int:
    def build_batch(cfg, args):
        x, y = make_mlm_batch(
            cfg.vocab_size,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            seed=args.seed if args.seed is not None else 0,
        )
        attn = torch.ones_like(x)
        # Contiguous token/position ids: HF buffer.expand() is non-contig
        # for batch>1 and nntile embedding rejects non-contiguous indices.
        token_type_ids = torch.zeros_like(x)
        position_ids = (
            torch.arange(args.seq_len, dtype=torch.long)
            .unsqueeze(0)
            .expand(args.batch_size, -1)
            .contiguous()
        )
        return {
            "input_ids": x,
            "labels": y,
            "attention_mask": attn,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
        }

    return run_tiny_hf_main(
        name="bert",
        argv=argv,
        default_config=_default_config(),
        config_cls=BertConfig,
        model_cls=BertForMaskedLM,
        build_batch=build_batch,
        loss_fn=mlm_ce_loss,
        attn_implementation="eager",
        description=__doc__,
    )


if __name__ == "__main__":
    raise SystemExit(main())
