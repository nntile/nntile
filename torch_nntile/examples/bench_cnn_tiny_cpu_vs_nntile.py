#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# Collect cpu vs nntile timings for tiny CNN train smokes (showcase).

"""Run tiny CNN train scripts on cpu and nntile; print a markdown table.

Example::

    python torch_nntile/examples/bench_cnn_tiny_cpu_vs_nntile.py \\
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

SCRIPTS = (
    ("lenet", "train_lenet_tiny.py"),
    ("resnet", "train_resnet_tiny.py"),
    ("vgg", "train_vgg_tiny.py"),
    ("mobilenet", "train_mobilenet_tiny.py"),
    ("unet", "train_unet_tiny.py"),
    ("unet_modern", "train_unet_modern_tiny.py"),
)

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
    name: str,
    script: str,
    device: str,
    ncpu: int,
    steps: int,
    batch_size: int,
    seed: int,
    output_root: Path,
) -> RunResult:
    out_dir = output_root / f"{name}_{device}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(here / script),
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
        capture_output=True,
        text=True,
        check=False,
        cwd=str(here.parent.parent),
    )
    elapsed = time.perf_counter() - t0
    text = proc.stdout + "\n" + proc.stderr
    loss = None
    wall = None
    losses = LOSS_RE.findall(text)
    if losses:
        loss = float(losses[-1])
    m = WALL_RE.search(text)
    if m:
        wall = float(m.group(1))
    ok = proc.returncode == 0 and loss is not None
    tail = ""
    if not ok:
        tail = (proc.stderr or proc.stdout or "")[-800:].strip()
    return RunResult(
        name=name,
        device=device,
        ok=ok,
        loss=loss,
        wall_s=wall if wall is not None else elapsed,
        elapsed_s=elapsed,
        stderr_tail=tail,
    )


def format_table(results: list[RunResult]) -> str:
    by_name: dict[str, dict[str, RunResult]] = {}
    for r in results:
        by_name.setdefault(r.name, {})[r.device] = r

    lines = [
        "| Model | CPU loss | nntile loss | CPU wall (s) | "
        "nntile wall (s) | Δ loss | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, _ in SCRIPTS:
        row = by_name.get(name, {})
        cpu = row.get("cpu")
        nnt = row.get("nntile")
        if cpu is None or nnt is None:
            lines.append(f"| {name} | — | — | — | — | — | incomplete |")
            continue
        if not cpu.ok or not nnt.ok:
            status = "FAIL"
            if not cpu.ok:
                status += " cpu"
            if not nnt.ok:
                status += " nntile"
            lines.append(
                f"| {name} | "
                f"{cpu.loss if cpu.loss is not None else '—'} | "
                f"{nnt.loss if nnt.loss is not None else '—'} | "
                f"{cpu.wall_s:.3f} | {nnt.wall_s:.3f} | — | {status} |"
            )
            continue
        assert cpu.loss is not None and nnt.loss is not None
        assert cpu.wall_s is not None and nnt.wall_s is not None
        dloss = abs(cpu.loss - nnt.loss)
        lines.append(
            f"| {name} | {cpu.loss:.6f} | {nnt.loss:.6f} | "
            f"{cpu.wall_s:.3f} | {nnt.wall_s:.3f} | {dloss:.3e} | OK |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ncpu", type=int, default=1)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output-root",
        default="/tmp/cnn_tiny_cpu_vs_nntile",
        help="Checkpoint / log root",
    )
    p.add_argument(
        "--markdown-out",
        default="",
        help="Optional path to write the markdown table",
    )
    args = p.parse_args(argv)

    here = Path(__file__).resolve().parent
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    for name, script in SCRIPTS:
        for device in ("cpu", "nntile"):
            print(
                f"==> {name} device={device} ncpu={args.ncpu}",
                flush=True,
            )
            r = run_one(
                here=here,
                name=name,
                script=script,
                device=device,
                ncpu=args.ncpu,
                steps=args.steps,
                batch_size=args.batch_size,
                seed=args.seed,
                output_root=output_root,
            )
            status = "OK" if r.ok else "FAIL"
            print(
                f"    {status} loss={r.loss} wall={r.wall_s} "
                f"elapsed={r.elapsed_s:.3f}s",
                flush=True,
            )
            if not r.ok and r.stderr_tail:
                print(f"    stderr_tail:\n{r.stderr_tail}", flush=True)
            results.append(r)

    table = format_table(results)
    print("\n" + table)
    if args.markdown_out:
        Path(args.markdown_out).write_text(table + "\n", encoding="utf-8")
        print(f"\nWrote {args.markdown_out}")

    failed = sum(1 for r in results if not r.ok)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
