#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_gpt_neox_hf.py
# Tiny stock HuggingFace GPTNeoXForCausalLM smoke on cpu/nntile.

"""Tiny HF GPT-NeoX causal LM smoke (synthetic tokens).

Example::

    python torch_nntile/examples/train_gpt_neox_hf.py --device nntile --steps 1
"""

from __future__ import annotations

import argparse

from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from hf_tiny_train_common import (
    add_common_args,
    causal_ce_loss,
    make_causal_batch,
    run_tiny_hf_train,
)


def tiny_config() -> GPTNeoXConfig:
    cfg = GPTNeoXConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        # Full rotary leaves an empty q_pass slice; TensorGraph rejects
        # zero-sized dims on backward. Partial RoPE keeps a pass-through.
        rotary_pct=0.5,
        rotary_emb_base=10000,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        classifier_dropout=0.0,
        bos_token_id=0,
        eos_token_id=0,
        tie_word_embeddings=False,
    )
    cfg._attn_implementation = "sdpa"
    cfg.use_cache = False
    return cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    args = p.parse_args(argv)
    cfg = tiny_config()

    def build_model():
        return GPTNeoXForCausalLM(cfg)

    def build_batch():
        x, y = make_causal_batch(
            cfg.vocab_size,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            seed=args.seed,
        )
        return {"input_ids": x, "labels": y}

    return run_tiny_hf_train(
        name="gpt-neox",
        args=args,
        build_model=build_model,
        build_batch=build_batch,
        loss_fn=causal_ce_loss,
    )


if __name__ == "__main__":
    raise SystemExit(main())
