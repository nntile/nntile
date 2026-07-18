#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/run_all_hf_tiny_smokes.py
# Run all tiny HF stock-model smokes and summarize failures.

"""Run gpt-neo / gpt-neox / llama / bert / roberta / t5 tiny HF smokes.

Example::

    python torch_nntile/examples/run_all_hf_tiny_smokes.py --device nntile
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = (
    "train_gpt_neo_hf.py",
    "train_gpt_neox_hf.py",
    "train_llama_hf.py",
    "train_bert_hf.py",
    "train_roberta_hf.py",
    "train_t5_hf.py",
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "nntile"), default="nntile")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--ncpu", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    here = Path(__file__).resolve().parent
    results: list[tuple[str, int]] = []
    for script in SCRIPTS:
        cmd = [
            sys.executable,
            str(here / script),
            "train",
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--steps",
            str(args.steps),
            "--seq-len",
            str(args.seq_len),
            "--batch-size",
            str(args.batch_size),
            "--ncpu",
            str(args.ncpu),
        ]
        print("\n" + "=" * 72)
        print(" ".join(cmd))
        print("=" * 72, flush=True)
        proc = subprocess.run(cmd, check=False)
        results.append((script, proc.returncode))

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    failed = 0
    for script, code in results:
        status = "OK" if code == 0 else f"FAIL({code})"
        print(f"  {status:10s}  {script}")
        if code != 0:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
