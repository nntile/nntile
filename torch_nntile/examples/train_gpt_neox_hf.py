#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_gpt_neox_hf.py
# Tiny stock HuggingFace GPTNeoXForCausalLM smoke on cpu/nntile.

"""Tiny HF GPT-NeoX causal LM smoke (synthetic tokens).

Uses JSON config / checkpoint like ``train_gpt2_hf.py``::

    python torch_nntile/examples/train_gpt_neox_hf.py train \\
        --device nntile --seed 0 --config gpt_neox_hf_tiny_config.json \\
        --output-dir /tmp/gpt_neox_hf --steps 1
"""

from __future__ import annotations

from pathlib import Path

from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from hf_tiny_train_common import (
    causal_ce_loss,
    make_causal_batch,
    run_tiny_hf_main,
)


def _default_config() -> Path:
    return Path(__file__).resolve().parent / "gpt_neox_hf_tiny_config.json"


def main(argv: list[str] | None = None) -> int:
    def build_batch(cfg, args):
        x, y = make_causal_batch(
            cfg.vocab_size,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            seed=args.seed if args.seed is not None else 0,
        )
        return {"input_ids": x, "labels": y}

    return run_tiny_hf_main(
        name="gpt-neox",
        argv=argv,
        default_config=_default_config(),
        config_cls=GPTNeoXConfig,
        model_cls=GPTNeoXForCausalLM,
        build_batch=build_batch,
        loss_fn=causal_ce_loss,
        attn_implementation="sdpa",
        use_cache=False,
        description=__doc__,
    )


if __name__ == "__main__":
    raise SystemExit(main())
