#!/usr/bin/env python3
"""Classic-kernel (torch_nntile.nn) overhead ladders. Does not re-run CUDA."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "torch_nntile" / "examples" / "train_nntile_native_overhead.py"

FAMILIES = {
    "gpt_neo": (
        REPO / "torch_nntile" / "examples" / "overhead_gpt_neo",
        {
            "xs": ("gpt_neo_xs.json", 768),
            "s": ("gpt_neo_s.json", 1024),
            "m": ("gpt_neo_m.json", 1536),
            "l": ("gpt_neo_l.json", 2048),
            "xl": ("gpt_neo_xl.json", 2880),
        },
    ),
    "gpt_neox": (
        REPO / "torch_nntile" / "examples" / "overhead_gpt_neox",
        {
            "xs": ("gpt_neox_xs.json", 768),
            "s": ("gpt_neox_s.json", 1024),
            "m": ("gpt_neox_m.json", 1536),
            "l": ("gpt_neox_l.json", 2048),
            "xl": ("gpt_neox_xl.json", 2880),
        },
    ),
    "llama": (
        REPO / "torch_nntile" / "examples" / "overhead_llama",
        {
            "xs": ("llama_xs.json", 768),
            "s": ("llama_s.json", 1024),
            "m": ("llama_m.json", 1536),
            "l": ("llama_l.json", 2048),
            "xl": ("llama_xl.json", 2560),
        },
    ),
    "bert": (
        REPO / "torch_nntile" / "examples" / "overhead_bert",
        {
            "xs": ("bert_xs.json", 768),
            "s": ("bert_s.json", 1024),
            "m": ("bert_m.json", 1536),
            "l": ("bert_l.json", 2048),
            "xl": ("bert_xl.json", 2880),
        },
    ),
    "roberta": (
        REPO / "torch_nntile" / "examples" / "overhead_roberta",
        {
            "xs": ("roberta_xs.json", 768),
            "s": ("roberta_s.json", 1024),
            "m": ("roberta_m.json", 1536),
            "l": ("roberta_l.json", 2048),
            "xl": ("roberta_xl.json", 2880),
        },
    ),
    "t5": (
        REPO / "torch_nntile" / "examples" / "overhead_t5",
        {
            "xs": ("t5_xs.json", 768),
            "s": ("t5_s.json", 1024),
            "m": ("t5_m.json", 1536),
            "l": ("t5_l.json", 2048),
            "xl": ("t5_xl.json", 2880),
        },
    ),
}


@dataclass
class RunResult:
    name: str
    family: str
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
    log_path: str = ""


def parse_log(text: str) -> RunResult:
    result = RunResult(name="", family="", size="", mode="")
    m = re.search(r"\[nntile\] final loss=([0-9.]+)", text)
    if m:
        result.final_loss = float(m.group(1))
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
    return result


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
    groups: dict[tuple[str, str], list[RunResult]] = defaultdict(list)
    for row in results:
        groups[(row.size, row.mode)].append(row)
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
        size, mode = key
        entry: dict[str, Any] = {
            "size": size,
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
        iso_keys: set[str] = set()
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
        if any(r.train_wall_s for r in rows):
            entry["metrics"]["host_s"] = _mean_std(
                [
                    (r.record_nntile_s or 0.0)
                    + (r.record_torch_s or 0.0)
                    + (r.compile_s or 0.0)
                    for r in rows
                    if r.train_wall_s is not None
                ]
            )
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
        summary["groups"][f"{size}_{mode}"] = entry
    return summary


def run_one(
    *,
    family: str,
    size: str,
    mode: str,
    logdir: Path,
    gpu: str,
    max_sequences: int = 10,
    repeat: int = 0,
    tag_repeat: bool = False,
) -> RunResult:
    config_dir, sizes = FAMILIES[family]
    cfg_name, seq_len = sizes[size]
    tag = f"{size}_nntile_native_{mode}"
    if tag_repeat:
        tag = f"{tag}_rep{repeat:02d}"
    out = logdir / tag
    out.mkdir(parents=True, exist_ok=True)
    log_path = logdir / f"{tag}.log"
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    env["CUDA_VISIBLE_DEVICES"] = gpu
    examples = str(REPO / "torch_nntile" / "examples")
    env["PYTHONPATH"] = f"{REPO / 'torch_nntile'}:{examples}"
    cmd = [
        sys.executable,
        "-u",
        str(TRAIN),
        "train",
        "--family",
        family,
        "--seed",
        "42",
        "--no-shuffle",
        "--config",
        str(config_dir / cfg_name),
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
        "--no-save-checkpoint",
        "--restrict-cuda",
        "--ncpu",
        "0",
        "--ncuda",
        "1",
    ]
    if mode == "sequential":
        cmd.append("--wait-after-run")
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
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
    parsed = parse_log(text)
    parsed.name = tag
    parsed.family = family
    parsed.size = size
    parsed.mode = mode
    parsed.repeat = repeat
    parsed.log_path = str(log_path)
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=list(FAMILIES))
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=None,
        help="Sizes (default: all for the family)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Also run --wait-after-run",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    _, sizes = FAMILIES[args.family]
    want = args.sizes if args.sizes else list(sizes)
    for size in want:
        if size not in sizes:
            raise SystemExit(f"unknown size {size} for {args.family}")
    args.logdir.mkdir(parents=True, exist_ok=True)
    results: list[RunResult] = []
    for repeat in range(args.repeats):
        print(
            f"=== {args.family} repeat {repeat + 1}/{args.repeats} ===",
            flush=True,
        )
        for size in want:
            print(
                f"run {args.family} {size} nntile_native overlap "
                f"rep={repeat}",
                flush=True,
            )
            results.append(
                run_one(
                    family=args.family,
                    size=size,
                    mode="overlap",
                    logdir=args.logdir,
                    gpu=args.gpu,
                    repeat=repeat,
                    tag_repeat=args.repeats > 1,
                )
            )
            if args.sequential:
                print(
                    f"run {args.family} {size} nntile_native sequential "
                    f"rep={repeat}",
                    flush=True,
                )
                results.append(
                    run_one(
                        family=args.family,
                        size=size,
                        mode="sequential",
                        logdir=args.logdir,
                        gpu=args.gpu,
                        repeat=repeat,
                        tag_repeat=args.repeats > 1,
                    )
                )
    out = args.logdir / "results.json"
    out.write_text(
        json.dumps([asdict(r) for r in results], indent=2),
        encoding="utf-8",
    )
    summary = summarize_results(results)
    summary["family"] = args.family
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
