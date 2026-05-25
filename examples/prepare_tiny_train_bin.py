#!/usr/bin/env python3
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file examples/prepare_tiny_train_bin.py
# Write a small uint16 token file for graph causal-LM training demos.
#
# @version 1.1.0

"""Create ``train.bin`` for ``--tiny`` Llama/GPT-2 graph training examples.

The C++ trainers mmap a flat ``uint16`` stream. Each batch consumes
``(seq_len + 1) * batch_size`` consecutive ids (input window + next-token
labels). Token ids stay in ``[0, vocab_size)`` (default 256 for ``--tiny``).

This script avoids HuggingFace downloads so the demo scripts run offline.
For real text data, use ``wrappers/python/examples/causal_lm_data_preparation.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_token_stream(
    num_tokens: int,
    vocab_size: int,
    seed: int,
) -> np.ndarray:
    """Low-entropy stream: slowly drifting ids (learnable next-token pattern)."""
    rng = np.random.default_rng(seed)
    tokens = np.empty(num_tokens, dtype=np.uint16)
    cur = int(rng.integers(0, vocab_size))
    step = int(rng.integers(1, max(2, vocab_size // 8)))
    for i in range(num_tokens):
        tokens[i] = np.uint16(cur % vocab_size)
        if rng.random() < 0.15:
            cur = int(rng.integers(0, vocab_size))
            step = int(rng.integers(1, max(2, vocab_size // 8)))
        else:
            cur += step
    return tokens


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a tiny uint16 train.bin for graph training demos",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path (e.g. build/examples/demo_data/llama/train.bin)",
    )
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--num-batches",
        type=int,
        default=8,
        help="How many full batches fit in the file",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=256,
        help="Must match --tiny config (default 256)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.seq_len < 1 or args.batch_size < 1 or args.num_batches < 1:
        raise SystemExit("seq-len, batch-size, and num-batches must be >= 1")
    if args.vocab_size < 2:
        raise SystemExit("vocab-size must be >= 2")

    tokens_per_batch = (args.seq_len + 1) * args.batch_size
    num_tokens = tokens_per_batch * args.num_batches
    tokens = build_token_stream(num_tokens, args.vocab_size, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(args.output)

    print(f"Wrote {args.output}")
    print(
        f"  tokens={num_tokens}  batches={args.num_batches}  "
        f"seq_len={args.seq_len}  batch_size={args.batch_size}  "
        f"vocab_size={args.vocab_size}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
