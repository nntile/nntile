#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_t5_hf.py
# Tiny stock HuggingFace T5ForConditionalGeneration smoke on cpu/nntile.

"""Tiny HF T5 encoder-decoder smoke (synthetic tokens).

Uses JSON config / checkpoint like ``train_gpt2_hf.py``::

    python torch_nntile/examples/train_t5_hf.py train \\
        --device nntile --seed 0 --config t5_hf_tiny_config.json \\
        --output-dir /tmp/t5_hf --steps 1
"""

from __future__ import annotations

from pathlib import Path

from transformers import T5Config, T5ForConditionalGeneration

from hf_tiny_train_common import (
    make_encoder_decoder_batch,
    run_tiny_hf_main,
    t5_ce_loss,
)


def _default_config() -> Path:
    return Path(__file__).resolve().parent / "t5_hf_tiny_config.json"


def main(argv: list[str] | None = None) -> int:
    def build_batch(cfg, args):
        enc, dec, labels = make_encoder_decoder_batch(
            cfg.vocab_size,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            seed=args.seed if args.seed is not None else 0,
        )
        return {
            "input_ids": enc,
            "decoder_input_ids": dec,
            "labels": labels,
        }

    return run_tiny_hf_main(
        name="t5",
        argv=argv,
        default_config=_default_config(),
        config_cls=T5Config,
        model_cls=T5ForConditionalGeneration,
        build_batch=build_batch,
        loss_fn=t5_ce_loss,
        attn_implementation="eager",
        use_cache=False,
        description=__doc__,
    )


if __name__ == "__main__":
    raise SystemExit(main())
