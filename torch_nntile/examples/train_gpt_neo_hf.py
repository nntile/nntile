#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_gpt_neo_hf.py
# Tiny stock HuggingFace GPTNeoForCausalLM smoke on cpu/nntile.

"""Tiny HF GPT-Neo causal LM smoke (synthetic tokens).

Example::

    python torch_nntile/examples/train_gpt_neo_hf.py --device nntile --steps 1
"""

from __future__ import annotations

import argparse

from transformers import GPTNeoConfig, GPTNeoForCausalLM

from hf_tiny_train_common import (
    add_common_args,
    causal_ce_loss,
    make_causal_batch,
    run_tiny_hf_train,
)


def tiny_config() -> GPTNeoConfig:
    cfg = GPTNeoConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=64,
        attention_types=[[["global"], 2]],
        window_size=32,
        attention_dropout=0.0,
        embed_dropout=0.0,
        resid_dropout=0.0,
        classifier_dropout=0.0,
        bos_token_id=0,
        eos_token_id=0,
    )
    # GPT-Neo has no SDPA path in this transformers version; eager uses
    # matmul attention (still exercises nntile mm/add/softmax/etc.).
    cfg._attn_implementation = "eager"
    cfg.use_cache = False
    return cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    args = p.parse_args(argv)
    cfg = tiny_config()

    def build_model():
        return GPTNeoForCausalLM(cfg)

    def build_batch():
        x, y = make_causal_batch(
            cfg.vocab_size,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            seed=args.seed,
        )
        return {"input_ids": x, "labels": y}

    return run_tiny_hf_train(
        name="gpt-neo",
        args=args,
        build_model=build_model,
        build_batch=build_batch,
        loss_fn=causal_ce_loss,
    )


if __name__ == "__main__":
    raise SystemExit(main())
