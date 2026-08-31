#!/usr/bin/env python3
"""Append GPT-2 XL ladder results to docs/dev/gpt2_hf_overhead_scale.md."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
from update_gpt2_overhead_doc import (  # noqa: E402
    DOC,
    g,
    iter_mean_table,
    ms,
    ms_s,
    pct,
)

REPO = Path(__file__).resolve().parents[2]


def _xl_rows(summary: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, str]:
    groups = summary["groups"]
    c = groups["xl_cuda_overlap"]
    n = groups["xl_nntile_overlap"]
    sq = groups["xl_nntile_sequential"]
    iso = n["isolated"]
    ratio = g(n, "metrics", "train_wall_s", "mean") / g(
        c, "metrics", "train_wall_s", "mean"
    )
    compute_mean = g(sq, "metrics", "run_s", "mean") + g(
        sq, "metrics", "wait_s", "mean"
    )
    compute_std = (
        g(sq, "metrics", "run_s", "std") ** 2
        + g(sq, "metrics", "wait_s", "std") ** 2
    ) ** 0.5
    compute_ratio = compute_mean / g(c, "metrics", "train_wall_s", "mean")
    prep_pct = (
        100
        * g(sq, "metrics", "host_s", "mean")
        / g(sq, "metrics", "train_wall_s", "mean")
    )
    full_iso = (
        g(iso, "record_nntile", "mean")
        + g(iso, "record_torch", "mean")
        + g(iso, "compile", "mean")
        + g(iso, "run", "mean")
        + g(iso, "wait", "mean")
    )
    rw = g(iso, "run_wait", "mean")
    saved = full_iso - rw
    pct_saved = 100 * saved / full_iso if full_iso else 0.0
    steady_vals = [
        r["iters"][1]["compute"]
        for r in results
        if r["size"] == "xl"
        and r["device"] == "nntile"
        and r["mode"] == "sequential"
        and len(r["iters"]) > 1
        and r["iters"][1].get("compute") is not None
    ]
    steady = statistics.mean(steady_vals) if steady_vals else 0.0
    return {
        "overall": (
            f"| XL T=2880 | {ms_s(g(c, 'metrics', 'train_wall_s'))} | "
            f"{ms_s(g(n, 'metrics', 'train_wall_s'))} | **{ratio:.2f}×** | "
            f"{ms_s(g(n, 'metrics', 'record_nntile_s'))} | "
            f"{ms_s(g(n, 'metrics', 'record_torch_s'))} | "
            f"{ms_s(g(n, 'metrics', 'compile_s'))} | "
            f"{ms_s(g(n, 'metrics', 'run_s'))} | "
            f"{ms_s(g(n, 'metrics', 'wait_s'))} | "
            f"{pct(g(n, 'metrics', 'host_frac'))} | — |"
        ),
        "isolated": (
            f"| XL | {ms(g(iso, 'record_nntile'))} | "
            f"{ms(g(iso, 'record_torch'))} | {ms(g(iso, 'compile'))} | "
            f"{ms(g(iso, 'run'))} | {ms(g(iso, 'wait'))} | "
            f"**{ms(g(iso, 'run_wait'))}** | "
            f"{ms(g(c['isolated'], 'cuda_wall'))} |"
        ),
        "hidden": (
            f"| XL | {full_iso:.3f} s | {rw:.3f} s | "
            f"{saved:.3f} s (**{pct_saved:.0f}%**) |"
        ),
        "sequential": (
            f"| XL T=2880 | {ms_s(g(c, 'metrics', 'train_wall_s'))} | "
            f"{ms_s(g(sq, 'metrics', 'train_wall_s'))} | "
            f"{ms_s(g(sq, 'metrics', 'host_s'))} | "
            f"**{ms({'mean': compute_mean, 'std': compute_std, 'n': 10})} s** | "
            f"**{compute_ratio:.2f}×** | {prep_pct:.1f}% |"
        ),
        "overlap_iter": iter_mean_table(results, "xl", "nntile", "overlap"),
        "sequential_iter": iter_mean_table(results, "xl", "nntile", "sequential"),
        "isolated_compare": (
            f"XL {ms(g(iso, 'run_wait'))} vs {ms(g(c['isolated'], 'cuda_wall'))} s"
        ),
        "steady": f"~{steady:.3f} s (XL)",
        "loss_note": f"XL {g(c, 'metrics', 'final_loss', 'mean'):.6f} both",
        "ratio": f"XL {ratio:.2f}×",
        "seq_ratio": f"{compute_ratio:.2f}×",
        "host_share": f"{g(n, 'metrics', 'host_frac', 'mean') * 100:.1f}%",
    }


def patch_doc(text: str, rows: dict[str, str]) -> str:
    text = text.replace(
        "Depth is **12 layers** everywhere.",
        "Depth is **12 layers** (XS–L); **XL** uses **6 layers** at similar param count.",
    )
    text = text.replace(
        "peaks at ~28 GiB CUDA / ~40 GiB nntile on a 46 GiB A40)",
        "peaks at ~28 GiB CUDA / ~40 GiB nntile on L; XL is ~26 GiB CUDA on a 46 GiB A40)",
    )
    text = text.replace("| | XS | S | M | L |", "| | XS | S | M | L | XL |")
    text = text.replace("|--|--:|--:|--:|--:|", "|--|--:|--:|--:|--:|--:|")
    text = text.replace(
        "| Config | `gpt2_xs.json` | `gpt2_s.json` | `gpt2_m.json` | `gpt2_l.json` |",
        "| Config | `gpt2_xs.json` | `gpt2_s.json` | `gpt2_m.json` | `gpt2_l.json` | `gpt2_xl.json` |",
    )
    text = text.replace(
        "| `n_layer` | 12 | 12 | 12 | 12 |",
        "| `n_layer` | 12 | 12 | 12 | 12 | **6** |",
    )
    text = text.replace(
        "| `n_embd` / `n_head` | 1536 / 24 | 2048 / 16 | 3072 / 24 | 4096 / 32 |",
        "| `n_embd` / `n_head` | 1536 / 24 | 2048 / 16 | 3072 / 24 | 4096 / 32 | 5760 / 45 |",
    )
    text = text.replace(
        "| `--seq-len` (`= n_embd/2`) | **768** | **1024** | **1536** | **2048** |",
        "| `--seq-len` (`= n_embd/2`) | **768** | **1024** | **1536** | **2048** | **2880** |",
    )
    text = text.replace(
        "| Params (FP32) | 344 M (1.28 GiB) | 611 M (2.44 GiB) | 1.37 B (5.49 GiB) | 2.44 B (9.74 GiB) |",
        "| Params (FP32) | 344 M (1.28 GiB) | 611 M (2.44 GiB) | 1.37 B (5.49 GiB) | 2.44 B (9.74 GiB) | **2.41 B (8.97 GiB)** |",
    )
    text = text.replace(
        "`device=nntile` `--ncpu 0 --ncuda 1 --restrict-cuda`. NVIDIA A40, **GPU 0**.",
        "`device=nntile` `--ncpu 0 --ncuda 1 --restrict-cuda`. NVIDIA A40, one GPU per job.",
    )
    text = text.replace(
        "Rerun: 2026-08-25, **10 repeats per configuration** (mean ± stdev;\n"
        "`/tmp/gpt2_overhead_x10_100step_20260825`).",
        "**10 repeats** per configuration (mean ± stdev).",
    )
    text = text.replace(
        "8.127417 both).",
        f"8.127417 both; {rows['loss_note']}).",
    )
    if rows["overall"] not in text:
        text = text.replace(
            "| L T=2048 | 18.953 ± 0.027 s | 17.843 ± 0.046 s | **0.94×** | 0.052 ± 0.002 s | 0.273 ± 0.006 s | 0.118 ± 0.005 s | 0.133 ± 0.004 s | 17.266 ± 0.049 s | **2.5%** | 28.2 / 40.4 GiB |",
            "| L T=2048 | 18.953 ± 0.027 s | 17.843 ± 0.046 s | **0.94×** | 0.052 ± 0.002 s | 0.273 ± 0.006 s | 0.118 ± 0.005 s | 0.133 ± 0.004 s | 17.266 ± 0.049 s | **2.5%** | 28.2 / 40.4 GiB |\n"
            + rows["overall"],
        )
    text = text.replace(
        "Host **share** drops **28.0% → 16.3% → 5.5% → 2.5%**",
        f"Host **share** drops **28.0% → 16.3% → 5.5% → 2.5% → {rows['host_share']}**",
    )
    text = text.replace(
        "L 1.759 ± 0.002 vs 1.878 ± 0.001 s).",
        f"L 1.759 ± 0.002 vs 1.878 ± 0.001 s, {rows['isolated_compare']}).",
    )
    if "### XL (`n_embd=5760`, `T=2880`)" not in text:
        text = text.replace(
            "### L (`n_embd=4096`, `T=2048`)\n\n| Iter | CUDA wall",
            "### L (`n_embd=4096`, `T=2048`)\n\n| Iter | CUDA wall",
        )
        insert = (
            "\n\n### XL (`n_embd=5760`, `T=2880`, 6 layers, `head_dim=128`)\n\n"
            "| Iter | CUDA wall | record(nntile) | record(torch) | compile | run | wait |\n"
            "|-----:|----------:|---------------:|--------------:|--------:|----:|-----:|\n"
            f"{rows['overlap_iter']}\n"
        )
        text = text.replace(
            "## Isolated extra step (mean ± stdev over 10 runs)",
            insert + "## Isolated extra step (mean ± stdev over 10 runs)",
        )
    if "| XL |" not in text.split("## Sequential prep")[0]:
        text = text.replace(
            "| L | 0.007 ± 0.000 | 0.042 ± 0.001 | 0.015 ± 0.001 | 0.014 ± 0.001 | 1.744 ± 0.002 | **1.759 ± 0.002** | 1.878 ± 0.001 |",
            "| L | 0.007 ± 0.000 | 0.042 ± 0.001 | 0.015 ± 0.001 | 0.014 ± 0.001 | 1.744 ± 0.002 | **1.759 ± 0.002** | 1.878 ± 0.001 |\n"
            + rows["isolated"],
        )
        text = text.replace(
            "| L | 1.823 s | 1.759 s | 0.064 s (**4%**) |",
            "| L | 1.823 s | 1.759 s | 0.064 s (**4%**) |\n" + rows["hidden"],
        )
    if "XL T=2880" not in text.split("## Sequential prep")[1].split("Loss matches")[0]:
        text = text.replace(
            "| L T=2048 | 18.953 ± 0.027 s | 18.241 ± 0.040 s | 0.485 ± 0.010 s | **17.754 ± 0.037 s** | **0.94×** | 2.7% |",
            "| L T=2048 | 18.953 ± 0.027 s | 18.241 ± 0.040 s | 0.485 ± 0.010 s | **17.754 ± 0.037 s** | **0.94×** | 2.7% |\n"
            + rows["sequential"],
        )
        text = text.replace(
            "Loss matches the overlapping runs (XS 7.888845, S 7.929048, M 7.996911, L 8.127417).",
            "Loss matches the overlapping runs (XS 7.888845, S 7.929048, M 7.996911, "
            f"L 8.127417, XL 8.389783).",
        )
        text = text.replace(
            "#### L (`T=2048`)\n\n| Iter | prep | compute",
            "#### L (`T=2048`)\n\n| Iter | prep | compute",
        )
        seq_insert = (
            "\n\n#### XL (`T=2880`)\n\n"
            "| Iter | prep | compute | record(nntile) | record(torch) | compile | run | wait |\n"
            "|-----:|-----:|--------:|---------------:|--------------:|--------:|----:|-----:|\n"
            f"{rows['sequential_iter']}\n"
        )
        text = text.replace(
            "Steady compute after iter 1 (mean over repeats):",
            seq_insert + "Steady compute after iter 1 (mean over repeats):",
        )
    text = text.replace(
        "~1.760 s (L).",
        f"~1.760 s (L), {rows['steady']}.",
    )
    text = text.replace(
        "1. **`seq_len = n_embd / 2`**: XS 768, S 1024, M 1536, L 2048.",
        "1. **`seq_len = n_embd / 2`**: XS 768, S 1024, M 1536, L 2048, XL 2880.",
    )
    text = text.replace(
        "Share **28.0% → 16.3% → 5.5% → 2.5%**.",
        f"Share **28.0% → 16.3% → 5.5% → 2.5% → {rows['host_share']}**.",
    )
    text = text.replace(
        "(XS 0.99×, S 0.96×, M 0.94×, L 0.94×).",
        f"(XS 0.99×, S 0.96×, M 0.94×, L 0.94×, {rows['ratio']}).",
    )
    text = text.replace(
        "**0.94× → 0.92× → 0.93× → 0.94×** CUDA.",
        f"**0.94× → 0.92× → 0.93× → 0.94× → {rows['seq_ratio']}** CUDA.",
    )
    if "--sizes xl" not in text:
        text = text.replace(
            "python3 torch_nntile/tools/run_gpt2_overhead_benchmark.py \\\n"
            "  --logdir /tmp/gpt2_overhead --gpu 0 --repeats 10",
            "python3 torch_nntile/tools/run_gpt2_overhead_benchmark.py \\\n"
            "  --logdir /tmp/gpt2_overhead --gpu 0 --repeats 10 --sizes xl --skip-long",
        )
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("/tmp/gpt2_overhead/results_summary.json"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("/tmp/gpt2_overhead/results.json"),
    )
    parser.add_argument("--output", type=Path, default=DOC)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    results = json.loads(args.results.read_text(encoding="utf-8"))
    rows = _xl_rows(summary, results)
    text = patch_doc(args.output.read_text(encoding="utf-8"), rows)
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
