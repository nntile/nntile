#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_t5_hf.py
# Tiny stock HuggingFace T5ForConditionalGeneration smoke on cpu/nntile.

"""Tiny HF T5 encoder-decoder smoke (synthetic tokens).

Example::

    python torch_nntile/examples/train_t5_hf.py --device nntile --steps 1
"""

from __future__ import annotations

import argparse

from transformers import T5Config, T5ForConditionalGeneration

from hf_tiny_train_common import (
    add_common_args,
    make_encoder_decoder_batch,
    run_tiny_hf_train,
    t5_ce_loss,
)


def tiny_config() -> T5Config:
    cfg = T5Config(
        vocab_size=128,
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        relative_attention_num_buckets=8,
        relative_attention_max_distance=32,
        dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    # T5 has no SDPA path here; eager relative-attention is the supported path.
    if hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = "eager"
    cfg.use_cache = False
    return cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    args = p.parse_args(argv)
    cfg = tiny_config()

    def build_model():
        return T5ForConditionalGeneration(cfg)

    def build_batch():
        enc, dec, labels = make_encoder_decoder_batch(
            cfg.vocab_size,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            seed=args.seed,
        )
        return {
            "input_ids": enc,
            "decoder_input_ids": dec,
            "labels": labels,
        }

    return run_tiny_hf_train(
        name="t5",
        args=args,
        build_model=build_model,
        build_batch=build_batch,
        loss_fn=t5_ce_loss,
    )


if __name__ == "__main__":
    raise SystemExit(main())
