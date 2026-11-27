#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# Collect cpu vs nntile timings for tiny DiT HF train smokes (showcase).

"""Run tiny DiT HF train on cpu and nntile; print a markdown table.

Example::

    python torch_nntile/examples/bench_dit_hf_tiny_cpu_vs_nntile.py \\
        --ncpu 1 --steps 1 --batch-size 2 --seed 0
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

LOSS_RE = re.compile(r"loss=([0-9.]+)")
WALL_RE = re.compile(r"wall=([0-9.]+)s")


@dataclass
class RunResult:
    name: str
    device: str
    ok: bool
    loss: float | None
    wall_s: float | None
    elapsed_s: float
    stderr_tail: str


def run_one(
    *,
    here: Path,
    device: str,
    ncpu: int,
    steps: int,
    batch_size: int,
    seed: int,
    output_root: Path,
) -> RunResult:
    out_dir = output_root / f"dit_{device}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(here / "train_dit_hf.py"),
        "train",
        "--device",
        device,
        "--seed",
        str(seed),
        "--steps",
        str(steps),
        "--batch-size",
        str(batch_size),
        "--ncpu",
        str(ncpu),
        "--output-dir",
        str(out_dir),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(here),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    loss_m = LOSS_RE.findall(text)
    wall_m = WALL_RE.findall(text)
    ok = proc.returncode == 0 and bool(loss_m) and "OK" in text
    return RunResult(
        name="dit",
        device=device,
        ok=ok,
        loss=float(loss_m[-1]) if loss_m else None,
        wall_s=float(wall_m[-1]) if wall_m else None,
        elapsed_s=elapsed,
        stderr_tail=(proc.stderr or "")[-800:],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncpu", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root",
        default="/tmp/dit_hf_tiny_cpu_vs_nntile",
    )
    parser.add_argument(
        "--markdown-out",
        default="",
        help="Optional path to write the markdown table",
    )
    args = parser.parse_args(argv)

    here = Path(__file__).resolve().parent
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    for device in ("cpu", "nntile"):
        print(f"==> dit {device}")
        r = run_one(
            here=here,
            device=device,
            ncpu=args.ncpu,
            steps=args.steps,
            batch_size=args.batch_size,
            seed=args.seed,
            output_root=output_root,
        )
        results.append(r)
        status = "OK" if r.ok else "FAIL"
        print(
            f"  {status} loss={r.loss} wall={r.wall_s} "
            f"elapsed={r.elapsed_s:.2f}s"
        )
        if not r.ok:
            print(r.stderr_tail)

    by_dev = {r.device: r for r in results}
    cpu = by_dev.get("cpu")
    nnt = by_dev.get("nntile")
    lines = [
        "| Model | CPU loss | nntile loss | CPU wall (s) | "
        "nntile wall (s) | Δ loss | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    if cpu and nnt and cpu.loss is not None and nnt.loss is not None:
        delta = abs(cpu.loss - nnt.loss)
        status = (
            "OK"
            if cpu.ok and nnt.ok and delta < 1e-5
            else "FAIL"
        )
        lines.append(
            f"| dit | {cpu.loss:.6f} | {nnt.loss:.6f} | "
            f"{cpu.wall_s if cpu.wall_s is not None else float('nan'):.3f} | "
            f"{nnt.wall_s if nnt.wall_s is not None else float('nan'):.3f} | "
            f"{delta:.3e} | {status} |"
        )
    else:
        lines.append("| dit | — | — | — | — | — | FAIL |")

    table = "\n".join(lines) + "\n"
    print(table)
    if args.markdown_out:
        Path(args.markdown_out).write_text(table, encoding="utf-8")
        print(f"Wrote {args.markdown_out}")

    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
