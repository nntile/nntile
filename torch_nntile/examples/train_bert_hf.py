#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_bert_hf.py
# Tiny stock HuggingFace BertForMaskedLM smoke on cpu/nntile.

"""Tiny HF BERT MLM smoke (synthetic tokens).

Example::

    python torch_nntile/examples/train_bert_hf.py --device nntile --steps 1
"""

from __future__ import annotations

import argparse

import torch
from transformers import BertConfig, BertForMaskedLM

from hf_tiny_train_common import (
    add_common_args,
    make_mlm_batch,
    mlm_ce_loss,
    run_tiny_hf_train,
)


def tiny_config() -> BertConfig:
    cfg = BertConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        type_vocab_size=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=0,
    )
    cfg._attn_implementation = "sdpa"
    return cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    args = p.parse_args(argv)
    cfg = tiny_config()

    def build_model():
        return BertForMaskedLM(cfg)

    def build_batch():
        x, y = make_mlm_batch(
            cfg.vocab_size,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            seed=args.seed,
        )
        attn = torch.ones_like(x)
        return {"input_ids": x, "labels": y, "attention_mask": attn}

    return run_tiny_hf_train(
        name="bert",
        args=args,
        build_model=build_model,
        build_batch=build_batch,
        loss_fn=mlm_ce_loss,
    )


if __name__ == "__main__":
    raise SystemExit(main())
