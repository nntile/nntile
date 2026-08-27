#!/usr/bin/env python3
"""Run GPT-2 HF overhead ladder and emit parsed JSON results.

All runs pin a single physical GPU via ``CUDA_VISIBLE_DEVICES`` (default
``--gpu 0``). GPT-Neo / GPT-NeoX overhead studies use the same GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "torch_nntile" / "examples" / "train_gpt2_hf.py"
CONFIG_DIR = REPO / "torch_nntile" / "examples" / "overhead_gpt2"

SIZES = {
    "xs": ("gpt2_xs.json", 768),
    "s": ("gpt2_s.json", 1024),
    "m": ("gpt2_m.json", 1536),
    "l": ("gpt2_l.json", 2048),
    "xl": ("gpt2_xl.json", 2880),
}
DEFAULT_LONG_STEPS = 100


@dataclass
class IterRow:
    step: int
    cuda_wall: float | None = None
    record_nntile: float | None = None
    record_torch: float | None = None
    compile: float | None = None
    run: float | None = None
    wait: float | None = None
    prep: float | None = None
    compute: float | None = None


@dataclass
class RunResult:
    name: str
    device: str
    size: str
    mode: str
    repeat: int = 0
    final_loss: float | None = None
    train_wall_s: float | None = None
    record_nntile_s: float | None = None
    record_torch_s: float | None = None
    compile_s: float | None = None
    run_s: float | None = None
    wait_s: float | None = None
    isolated: dict[str, float] = field(default_factory=dict)
    peak_vram_gib: float | None = None
    iters: list[IterRow] = field(default_factory=list)
    log_path: str = ""


def parse_log(text: str, device: str) -> RunResult:
    result = RunResult(name="", device=device, size="", mode="")
    m = re.search(
        r"\[(cuda|nntile)\] final loss=([0-9.]+)",
        text,
    )
    if m:
        result.final_loss = float(m.group(2))
    if device == "cuda":
        m = re.search(
            r"timing torch train wall \(loop\+sync, loss readout after\): "
            r"([0-9.]+)s",
            text,
        )
        if m:
            result.train_wall_s = float(m.group(1))
        for line in text.splitlines():
            m = re.search(
                r"timing torch iter (\d+)/\d+ wall=([0-9.]+)s",
                line,
            )
            if m:
                result.iters.append(
                    IterRow(step=int(m.group(1)), cuda_wall=float(m.group(2)))
                )
        m = re.search(r"timing torch isolated iter wall=([0-9.]+)s", text)
        if m:
            result.isolated["cuda_wall"] = float(m.group(1))
    else:
        m = re.search(
            r"timing nntile train wall "
            r"\(loop through final wait, loss readout after\): "
            r"([0-9.]+)s",
            text,
        )
        if m:
            result.train_wall_s = float(m.group(1))
        for key, pat in [
            ("record_nntile_s", r"timing nntile record\(nntile\): ([0-9.]+)s"),
            ("record_torch_s", r"timing nntile record\(torch\): ([0-9.]+)s"),
            ("compile_s", r"timing nntile compile: ([0-9.]+)s"),
            ("run_s", r"timing nntile run: ([0-9.]+)s"),
            ("wait_s", r"timing nntile wait: ([0-9.]+)s"),
        ]:
            m = re.search(pat, text)
            if m:
                setattr(result, key, float(m.group(1)))
        for line in text.splitlines():
            m = re.search(
                r"timing nntile iter (\d+)/\d+ "
                r"record\(nntile\)=([0-9.]+)s "
                r"record\(torch\)=([0-9.]+)s "
                r"compile=([0-9.]+)s run=([0-9.]+)s "
                r"wait=([0-9.]+)s"
                r"(?: prep=([0-9.]+)s compute=([0-9.]+)s)?",
                line,
            )
            if m:
                row = IterRow(
                    step=int(m.group(1)),
                    record_nntile=float(m.group(2)),
                    record_torch=float(m.group(3)),
                    compile=float(m.group(4)),
                    run=float(m.group(5)),
                    wait=float(m.group(6)),
                )
                if m.group(7):
                    row.prep = float(m.group(7))
                    row.compute = float(m.group(8))
                result.iters.append(row)
        m = re.search(
            r"timing nntile isolated "
            r"record\(nntile\)=([0-9.]+)s "
            r"record\(torch\)=([0-9.]+)s "
            r"compile=([0-9.]+)s run=([0-9.]+)s "
            r"wait=([0-9.]+)s run\+wait=([0-9.]+)s",
            text,
        )
        if m:
            result.isolated = {
                "record_nntile": float(m.group(1)),
                "record_torch": float(m.group(2)),
                "compile": float(m.group(3)),
                "run": float(m.group(4)),
                "wait": float(m.group(5)),
                "run_wait": float(m.group(6)),
            }
    m = re.search(r"peak_vram_gib=([0-9.]+)", text)
    if m:
        result.peak_vram_gib = float(m.group(1))
    return result


def run_one(
    *,
    size: str,
    device: str,
    mode: str,
    logdir: Path,
    max_sequences: int = 10,
    gpu: str = "0",
    repeat: int = 0,
) -> RunResult:
    cfg_name, seq_len = SIZES[size]
    cfg = CONFIG_DIR / cfg_name
    tag = f"{size}_{device}_{mode}"
    if max_sequences != 10:
        tag = f"{size}_nntile_{max_sequences}step"
    if repeat > 0:
        tag = f"{tag}_rep{repeat:02d}"
    out = logdir / tag
    out.mkdir(parents=True, exist_ok=True)
    log_path = logdir / f"{tag}.log"

    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    env["CUDA_VISIBLE_DEVICES"] = gpu
    examples_path = str(REPO / "torch_nntile" / "examples")
    if device == "cuda":
        env["PYTHONPATH"] = examples_path
    else:
        env["PYTHONPATH"] = f"{REPO / 'torch_nntile'}:{examples_path}"
    cmd = [
        sys.executable,
        "-u",
        str(TRAIN),
        "train",
        "--seed",
        "42",
        "--no-shuffle",
        "--config",
        str(cfg),
        "--seq-len",
        str(seq_len),
        "--batch-size",
        "1",
        "--max-sequences",
        str(max_sequences),
        "--epochs",
        "1",
        "--output-dir",
        str(out),
    ]
    cmd.append("--no-save-checkpoint")
    insert_at = 4
    if device == "cuda":
        cmd[insert_at:insert_at] = ["--device", "cuda", "--disable-tf32"]
    else:
        cmd[insert_at:insert_at] = [
            "--device",
            "nntile",
            "--restrict-cuda",
            "--ncpu",
            "0",
            "--ncuda",
            "1",
        ]
        if mode == "sequential":
            cmd.append("--wait-after-run")

    exec_cmd = cmd

    t0 = time.perf_counter()
    proc = subprocess.run(
        exec_cmd,
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    text = proc.stdout + "\n" + proc.stderr
    log_path.write_text(text, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"run failed {tag} rc={proc.returncode} elapsed={elapsed:.1f}s\n"
            f"{text[-4000:]}"
        )
    parsed = parse_log(text, device)
    parsed.name = tag
    parsed.size = size
    parsed.mode = mode
    parsed.repeat = repeat
    parsed.log_path = str(log_path)
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    return parsed


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    if len(values) == 1:
        return {
            "mean": values[0],
            "std": 0.0,
            "min": values[0],
            "max": values[0],
            "n": 1,
        }
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def summarize_results(results: list[RunResult]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[RunResult]] = defaultdict(list)
    for row in results:
        groups[(row.size, row.device, row.mode)].append(row)

    summary: dict[str, Any] = {"groups": {}, "repeats": 0}
    if results:
        summary["repeats"] = len({r.repeat for r in results})

    scalar_fields = [
        "final_loss",
        "train_wall_s",
        "record_nntile_s",
        "record_torch_s",
        "compile_s",
        "run_s",
        "wait_s",
    ]
    for key, rows in sorted(groups.items()):
        size, device, mode = key
        entry: dict[str, Any] = {
            "size": size,
            "device": device,
            "mode": mode,
            "n": len(rows),
            "metrics": {},
        }
        for field_name in scalar_fields:
            vals = [
                float(getattr(r, field_name))
                for r in rows
                if getattr(r, field_name) is not None
            ]
            if vals:
                entry["metrics"][field_name] = _mean_std(vals)
        iso_keys = set()
        for r in rows:
            iso_keys.update(r.isolated.keys())
        if iso_keys:
            entry["isolated"] = {}
            for iso_key in sorted(iso_keys):
                vals = [
                    float(r.isolated[iso_key])
                    for r in rows
                    if iso_key in r.isolated
                ]
                if vals:
                    entry["isolated"][iso_key] = _mean_std(vals)
        entry["metrics"]["host_s"] = _mean_std(
            [
                (r.record_nntile_s or 0.0)
                + (r.record_torch_s or 0.0)
                + (r.compile_s or 0.0)
                for r in rows
                if r.train_wall_s is not None
            ]
        )
        if any(r.train_wall_s for r in rows):
            entry["metrics"]["host_frac"] = _mean_std(
                [
                    (
                        (r.record_nntile_s or 0.0)
                        + (r.record_torch_s or 0.0)
                        + (r.compile_s or 0.0)
                    )
                    / r.train_wall_s
                    for r in rows
                    if r.train_wall_s is not None and r.train_wall_s > 0
                ]
            )
        summary["groups"][f"{size}_{device}_{mode}"] = entry
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--long-steps",
        type=int,
        default=DEFAULT_LONG_STEPS,
        help="S nntile steady-state run length (default: 100)",
    )
    parser.add_argument(
        "--skip-long",
        action="store_true",
        help="Skip the long S nntile run",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=list(SIZES),
        default=list(SIZES),
        help="Ladder sizes to run (default: all)",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.long_steps < 1:
        raise SystemExit("--long-steps must be >= 1")
    args.logdir.mkdir(parents=True, exist_ok=True)
    results: list[RunResult] = []
    for repeat in range(args.repeats):
        print(f"=== repeat {repeat + 1}/{args.repeats} ===", flush=True)
        for size in args.sizes:
            for device in ("cuda", "nntile"):
                print(
                    f"run {size} {device} overlap rep={repeat}",
                    flush=True,
                )
                results.append(
                    run_one(
                        size=size,
                        device=device,
                        mode="overlap",
                        logdir=args.logdir,
                        gpu=args.gpu,
                        repeat=repeat,
                    )
                )
            print(f"run {size} nntile sequential rep={repeat}", flush=True)
            results.append(
                run_one(
                    size=size,
                    device="nntile",
                    mode="sequential",
                    logdir=args.logdir,
                    gpu=args.gpu,
                    repeat=repeat,
                )
            )
        if not args.skip_long and "s" in args.sizes:
            long_mode = f"{args.long_steps}step"
            print(
                f"run s nntile {long_mode} rep={repeat}",
                flush=True,
            )
            results.append(
                run_one(
                    size="s",
                    device="nntile",
                    mode=long_mode,
                    logdir=args.logdir,
                    max_sequences=args.long_steps,
                    gpu=args.gpu,
                    repeat=repeat,
                )
            )
    out = args.logdir / "results.json"
    out.write_text(
        json.dumps([asdict(r) for r in results], indent=2),
        encoding="utf-8",
    )
    summary = summarize_results(results)
    summary["long_steps"] = args.long_steps
    summary_path = args.logdir / "results_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
