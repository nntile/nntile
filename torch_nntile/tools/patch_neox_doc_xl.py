#!/usr/bin/env python3
"""Append XL ladder results to docs/dev/gpt_neox_hf_overhead_scale.md."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
from update_gpt_neox_overhead_doc import g, iter_mean_table, ms, ms_s, pct  # noqa: E402

DOC = Path(__file__).resolve().parents[2] / "docs" / "dev" / "gpt_neox_hf_overhead_scale.md"


def grp(summary: dict[str, Any], size: str, device: str, mode: str) -> dict:
    return summary["groups"][f"{size}_{device}_{mode}"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    results = json.loads(args.results.read_text(encoding="utf-8"))
    repeats = summary.get("repeats", 1)

    c = grp(summary, "xl", "cuda", "overlap")
    n = grp(summary, "xl", "nntile", "overlap")
    ratio = g(n, "metrics", "train_wall_s", "mean") / g(
        c, "metrics", "train_wall_s", "mean"
    )
    c_loss = g(c, "metrics", "final_loss", "mean")
    n_loss = g(n, "metrics", "final_loss", "mean")
    loss_cell = f"{c_loss:.6f} | **{n_loss:.6f}**"
    overall_row = (
        f"| XL T=2880 | {ms_s(g(c, 'metrics', 'train_wall_s'))} | "
        f"{ms_s(g(n, 'metrics', 'train_wall_s'))} | **{ratio:.2f}×** | "
        f"{ms_s(g(n, 'metrics', 'record_nntile_s'))} | "
        f"{ms_s(g(n, 'metrics', 'record_torch_s'))} | "
        f"{ms_s(g(n, 'metrics', 'compile_s'))} | "
        f"{ms_s(g(n, 'metrics', 'run_s'))} | "
        f"{ms_s(g(n, 'metrics', 'wait_s'))} | "
        f"{pct(g(n, 'metrics', 'host_frac'))} | {loss_cell} |"
    )

    iso_n = grp(summary, "xl", "nntile", "overlap")["isolated"]
    iso_c = grp(summary, "xl", "cuda", "overlap")["isolated"]
    iso_line = (
        f"XL {ms(g(iso_n, 'run_wait'))} vs {ms(g(iso_c, 'cuda_wall'))} s"
    )

    text = DOC.read_text(encoding="utf-8")

    if "| XL T=2880 |" in text:
        text = re.sub(
            r"\| XL T=2880 \|[^\n]+\n",
            overall_row + "\n",
            text,
            count=1,
        )
    else:
        text = text.replace(
            "| L T=2048 |",
            "| L T=2048 |",
        )
        text = re.sub(
            r"(\| L T=2048 \|[^\n]+\n)",
            r"\1" + overall_row + "\n",
            text,
            count=1,
        )

    if "`gpt_neox_xl.json`" not in text:
        text = text.replace(
            "| `gpt_neox_l.json` |",
            "| `gpt_neox_l.json` | `gpt_neox_xl.json` |",
        )
        text = text.replace(
            "| 12 | 12 | 12 | 12 |",
            "| 12 | 12 | 12 | 12 | **6** |",
        )
        text = text.replace(
            "| 4096 / 32 |",
            "| 4096 / 32 | **5760 / 45** |",
        )
        text = text.replace(
            "| **2048** |",
            "| **2048** | **2880** |",
            1,
        )
        text = text.replace(
            "| 2.43 B (9.06 GiB) |",
            "| 2.43 B (9.06 GiB) | **2.41 B (8.97 GiB)** |",
        )

    if "2.8%" in text and "→ 2.8%" in text:
        host_xl = f"{g(n, 'metrics', 'host_frac', 'mean') * 100:.1f}%"
        text = text.replace("→ 2.8%**", f"→ 2.8% → {host_xl}**")

    if "L 1.795" in text or "L " in text and "isolated" in text:
        if ",\nXL " not in text and "XL " not in text.split("Isolated GPU")[1][:200]:
            text = re.sub(
                r"(L [0-9.]+ ± [0-9.]+ vs [0-9.]+ ± [0-9.]+ s)\.",
                r"\1,\n" + iso_line + ".",
                text,
                count=1,
            )

    xl_iter = f"""
### XL (`hidden_size=5760`, `T=2880`, 6 layers, `head_dim=128`)

| Iter | CUDA wall | record(nntile) | record(torch) | compile | run | wait |
|-----:|----------:|---------------:|--------------:|--------:|----:|-----:|
{iter_mean_table(results, 'xl', 'nntile', 'overlap')}
"""
    if "### XL (`hidden_size=5760`" not in text:
        text = text.replace(
            "## Isolated extra step",
            xl_iter + "\n## Isolated extra step",
        )
    else:
        text = re.sub(
            r"### XL \(`hidden_size=5760`[^\n]*\n\n\| Iter \| CUDA wall \|[^\n]+\n\|-----:[^\n]+\n(?:\|[^\n]+\n)+",
            xl_iter.strip() + "\n",
            text,
            count=1,
        )

    if "**1.00×**" in text and "XL" not in text.split("Conclusions")[0][-500:]:
        text = text.replace(
            "L **1.00×**",
            f"L **1.00×**, XL **{ratio:.2f}×**",
            1,
        )

    DOC.write_text(text, encoding="utf-8")
    print(f"patched {DOC}")
    print(f"XL nntile/CUDA: {ratio:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
