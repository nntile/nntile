#!/usr/bin/env python3
"""Regenerate docs/dev/llama_hf_overhead_scale.md from benchmark summary."""

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
from overhead_plot import write_long_plots
from overhead_refs import GPT2_REF, GPT_NEOX_REF

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "dev" / "llama_hf_overhead_scale.md"

SIZE_LABEL = {"xs": "XS", "s": "S", "m": "M", "l": "L", "xl": "XL"}
LADDER = ("xs", "s", "m", "l", "xl")
SEQ_LEN = {"xs": 768, "s": 1024, "m": 1536, "l": 2048, "xl": 2880}
HIDDEN = {
    "xs": "1536 / 24",
    "s": "2048 / 16",
    "m": "3072 / 24",
    "l": "4096 / 32",
    "xl": "5760 / 45",
}
LAYERS = {"xs": 12, "s": 12, "m": 12, "l": 12, "xl": 6}
PARAMS = {
    "xs": "344 M (1.28 GiB)",
    "s": "611 M (2.27 GiB)",
    "m": "1.37 B (5.10 GiB)",
    "l": "2.43 B (9.06 GiB)",
    "xl": "2.41 B (8.96 GiB)",
}


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_summaries(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged["groups"] = {
        **base.get("groups", {}),
        **extra.get("groups", {}),
    }
    merged["repeats"] = max(
        int(base.get("repeats", 0)),
        int(extra.get("repeats", 0)),
    )
    merged["long_steps"] = int(
        base.get("long_steps", extra.get("long_steps", 100))
    )
    return merged


def g(group: dict[str, Any], *keys: str) -> dict[str, float]:
    cur: Any = group
    for key in keys:
        cur = cur[key]
    return cur


def ms(stat: dict[str, float], ndigits: int = 3) -> str:
    mean = stat["mean"]
    std = stat["std"]
    if stat["n"] <= 1 or std == 0:
        return f"{mean:.{ndigits}f}"
    return f"{mean:.{ndigits}f} ± {std:.{ndigits}f}"


def ms_s(stat: dict[str, float], ndigits: int = 3) -> str:
    return f"{ms(stat, ndigits)} s"


def pct(stat: dict[str, float]) -> str:
    return f"**{stat['mean'] * 100:.1f}%**"


def iter_mean_table(
    results: list[dict[str, Any]],
    size: str,
    device: str,
    mode: str,
) -> str:
    rows = [
        r
        for r in results
        if r["size"] == size and r["device"] == device and r["mode"] == mode
    ]
    if not rows:
        return ""
    n_iters = len(rows[0].get("iters", []))
    lines = []
    for step in range(n_iters):
        if device == "cuda":
            vals = [
                r["iters"][step]["cuda_wall"]
                for r in rows
                if len(r["iters"]) > step
            ]
            stat = {
                "mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
            lines.append(f"| {step + 1} | {ms(stat)} |")
        else:
            cols: dict[str, list[float]] = {
                "record_nntile": [],
                "record_torch": [],
                "compile": [],
                "run": [],
                "wait": [],
            }
            if mode == "sequential":
                cols["prep"] = []
                cols["compute"] = []
            for r in rows:
                if len(r["iters"]) <= step:
                    continue
                it = r["iters"][step]
                for key in cols:
                    if key in it and it[key] is not None:
                        cols[key].append(float(it[key]))
            if mode == "sequential":
                parts = [str(step + 1)]
                for key in [
                    "prep",
                    "compute",
                    "record_nntile",
                    "record_torch",
                    "compile",
                    "run",
                    "wait",
                ]:
                    vals = cols.get(key, [])
                    if not vals:
                        parts.append("—")
                    else:
                        stat = {
                            "mean": statistics.mean(vals),
                            "std": (
                                statistics.stdev(vals) if len(vals) > 1 else 0.0
                            ),
                            "n": len(vals),
                        }
                        parts.append(ms(stat))
                lines.append("| " + " | ".join(parts) + " |")
            else:
                cuda_rows = [
                    r
                    for r in results
                    if r["size"] == size
                    and r["device"] == "cuda"
                    and r["mode"] == "overlap"
                ]
                cuda_vals = [
                    r["iters"][step]["cuda_wall"]
                    for r in cuda_rows
                    if len(r["iters"]) > step
                ]
                cuda_stat = {
                    "mean": statistics.mean(cuda_vals),
                    "std": (
                        statistics.stdev(cuda_vals) if len(cuda_vals) > 1 else 0.0
                    ),
                    "n": len(cuda_vals),
                }
                parts = [str(step + 1), ms(cuda_stat)]
                for key in [
                    "record_nntile",
                    "record_torch",
                    "compile",
                    "run",
                    "wait",
                ]:
                    vals = cols[key]
                    stat = {
                        "mean": statistics.mean(vals),
                        "std": (
                            statistics.stdev(vals) if len(vals) > 1 else 0.0
                        ),
                        "n": len(vals),
                    }
                    parts.append(ms(stat))
                lines.append("| " + " | ".join(parts) + " |")
    return "\n".join(lines)


def render_doc(
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    logdir: str,
    preliminary_note: str = "",
) -> str:
    groups = summary["groups"]
    repeats = summary["repeats"]
    long_steps = int(summary.get("long_steps", 100))
    long_mode = f"{long_steps}step"
    prelim_block = ""
    if preliminary_note:
        prelim_block = (
            f"> **Preliminary ({preliminary_note}).** "
            "Full 10× rerun with `--long-steps 100` is in progress.\n\n"
        )

    def grp(size: str, device: str, mode: str) -> dict[str, Any]:
        return groups[f"{size}_{device}_{mode}"]

    overall_rows = []
    host_shares = []
    ratios = []
    loss_notes = []
    for size in LADDER:
        c = grp(size, "cuda", "overlap")
        n = grp(size, "nntile", "overlap")
        ratio_mean = g(n, "metrics", "train_wall_s", "mean") / g(
            c, "metrics", "train_wall_s", "mean"
        )
        c_loss = g(c, "metrics", "final_loss", "mean")
        n_loss = g(n, "metrics", "final_loss", "mean")
        loss_cell = (
            f"{c_loss:.6f} | **{n_loss:.6f}**"
            if abs(c_loss - n_loss) < 1e-5
            else f"{c_loss:.6f} | **{n_loss:.6f}**"
        )
        if abs(c_loss - n_loss) >= 1e-5:
            loss_notes.append(
                f"- **{SIZE_LABEL[size]}:** CUDA {c_loss:.6f} vs nntile "
                f"{n_loss:.6f} (Δ {abs(c_loss - n_loss):.6f})."
            )
        overall_rows.append(
            f"| {SIZE_LABEL[size]} T={SEQ_LEN[size]} | {ms_s(g(c, 'metrics', 'train_wall_s'))} | "
            f"{ms_s(g(n, 'metrics', 'train_wall_s'))} | **{ratio_mean:.2f}×** | "
            f"{ms_s(g(n, 'metrics', 'record_nntile_s'))} | "
            f"{ms_s(g(n, 'metrics', 'record_torch_s'))} | "
            f"{ms_s(g(n, 'metrics', 'compile_s'))} | "
            f"{ms_s(g(n, 'metrics', 'run_s'))} | "
            f"{ms_s(g(n, 'metrics', 'wait_s'))} | "
            f"{pct(g(n, 'metrics', 'host_frac'))} | {loss_cell} |"
        )
        host_shares.append(f"{g(n, 'metrics', 'host_frac', 'mean') * 100:.1f}%")
        ratios.append(f"{SIZE_LABEL[size]} {ratio_mean:.2f}×")

    xs_c_loss = g(grp("xs", "cuda", "overlap"), "metrics", "final_loss", "mean")
    xs_n_loss = g(grp("xs", "nntile", "overlap"), "metrics", "final_loss", "mean")
    if loss_notes:
        loss_section = "### Loss / correctness\n\n" + "\n".join(loss_notes)
        if abs(xs_c_loss - xs_n_loss) < 1e-5:
            loss_section = (
                f"- **XS:** loss matches to printed digits "
                f"({xs_c_loss:.6f}).\n\n" + loss_section
            )
        loss_section += (
            "\n\nPerformance ratios remain informative; investigate eager-attention "
            "training parity separately from graph overhead."
        )
    else:
        loss_section = (
            f"Loss matches CUDA vs nntile to printed 1e-5 "
            f"(XS {xs_c_loss:.6f} both)."
        )

    iso_rows = []
    hidden_rows = []
    for size in LADDER:
        n = grp(size, "nntile", "overlap")
        c = grp(size, "cuda", "overlap")
        iso = n["isolated"]
        full_mean = (
            g(iso, "record_nntile", "mean")
            + g(iso, "record_torch", "mean")
            + g(iso, "compile", "mean")
            + g(iso, "run", "mean")
            + g(iso, "wait", "mean")
        )
        rw = g(iso, "run_wait", "mean")
        saved = full_mean - rw
        pct_saved = 100 * saved / full_mean if full_mean else 0
        iso_rows.append(
            f"| {SIZE_LABEL[size]} | {ms(g(iso, 'record_nntile'))} | "
            f"{ms(g(iso, 'record_torch'))} | {ms(g(iso, 'compile'))} | "
            f"{ms(g(iso, 'run'))} | {ms(g(iso, 'wait'))} | "
            f"**{ms(g(iso, 'run_wait'))}** | "
            f"{ms(g(c['isolated'], 'cuda_wall'))} |"
        )
        hidden_rows.append(
            f"| {SIZE_LABEL[size]} | {full_mean:.3f} s | {rw:.3f} s | "
            f"{saved:.3f} s (**{pct_saved:.0f}%**) |"
        )

    seq_rows = []
    seq_loss = []
    seq_compute_ratios = []
    for size in LADDER:
        c = grp(size, "cuda", "overlap")
        sq = grp(size, "nntile", "sequential")
        prep = g(sq, "metrics", "host_s")
        compute = {
            "mean": g(sq, "metrics", "run_s", "mean")
            + g(sq, "metrics", "wait_s", "mean"),
            "std": (
                g(sq, "metrics", "run_s", "std") ** 2
                + g(sq, "metrics", "wait_s", "std") ** 2
            )
            ** 0.5,
            "n": repeats,
        }
        compute_ratio = compute["mean"] / g(c, "metrics", "train_wall_s", "mean")
        prep_pct = 100 * prep["mean"] / g(sq, "metrics", "train_wall_s", "mean")
        seq_rows.append(
            f"| {SIZE_LABEL[size]} T={SEQ_LEN[size]} | {ms_s(g(c, 'metrics', 'train_wall_s'))} | "
            f"{ms_s(g(sq, 'metrics', 'train_wall_s'))} | {ms_s(prep)} | "
            f"**{ms(compute)} s** | **{compute_ratio:.2f}×** | {prep_pct:.1f}% |"
        )
        seq_loss.append(
            f"{SIZE_LABEL[size]} {g(sq, 'metrics', 'final_loss', 'mean'):.6f}"
        )
        seq_compute_ratios.append(f"{compute_ratio:.2f}×")

    s1k = groups.get(f"s_nntile_{long_mode}")
    compare_long = ""
    if s1k:
        host1k = (
            100
            * (
                g(s1k, "metrics", "record_nntile_s", "mean")
                + g(s1k, "metrics", "record_torch_s", "mean")
                + g(s1k, "metrics", "compile_s", "mean")
            )
            / g(s1k, "metrics", "train_wall_s", "mean")
        )
        llama_long_wall = g(s1k, "metrics", "train_wall_s", "mean")
        llama_long_loss = g(s1k, "metrics", "final_loss", "mean")
        compare_long = f"""
### {long_steps}-step S (nntile)

| | GPT-2 | GPT-NeoX | Llama | Notes |
|--|------:|---------:|------:|-------|
| train wall | {GPT2_REF['long_wall_s']:.1f} s | {GPT_NEOX_REF['long_wall_s']:.1f} s | **{llama_long_wall:.1f} s** | same ballpark |
| final loss | {GPT2_REF['long_loss']:.6f} | {GPT_NEOX_REF['long_loss']:.6f} | **{llama_long_loss:.6f}** | see 10-step loss table |
| host share | {GPT2_REF['long_host_pct']}% | {GPT_NEOX_REF['long_host_pct']}% | **{host1k:.0f}%** | flat host, GPU-bound |"""
        long_section = f"""## {long_steps}-step S (nntile steady state, mean ± stdev over {repeats} runs)

Same **S** config (`hidden_size=2048`, `T=1024`, B=1), **{long_steps} optimizer steps**, nntile
overlap only. Complements the 10-step ladder above.

Loss **{llama_long_loss:.6f}**.

| | Total | mean / step |
|--|--:|--:|
| record(nntile) | {ms_s(g(s1k, 'metrics', 'record_nntile_s'))} | {g(s1k, 'metrics', 'record_nntile_s', 'mean') / long_steps * 1000:.1f} ms |
| record(torch) | {ms_s(g(s1k, 'metrics', 'record_torch_s'))} | {g(s1k, 'metrics', 'record_torch_s', 'mean') / long_steps * 1000:.0f} ms |
| compile | {ms_s(g(s1k, 'metrics', 'compile_s'))} | {g(s1k, 'metrics', 'compile_s', 'mean') / long_steps * 1000:.0f} ms |
| run | {ms_s(g(s1k, 'metrics', 'run_s'))} | {g(s1k, 'metrics', 'run_s', 'mean') / long_steps * 1000:.0f} ms |
| wait | {ms_s(g(s1k, 'metrics', 'wait_s'))} | {g(s1k, 'metrics', 'wait_s', 'mean') / long_steps * 1000:.0f} ms |
| **train wall** | **{ms_s(g(s1k, 'metrics', 'train_wall_s'))}** | {g(s1k, 'metrics', 'train_wall_s', 'mean') / long_steps * 1000:.0f} ms |

Host (record + compile) is **{host1k:.0f}%** of the wall (~{g(s1k, 'metrics', 'host_s', 'mean') / long_steps * 1000:.0f} ms/step).

![Host overhead per iteration](llama_hf_overhead_s_{long_steps}.svg)

CSV: [`llama_hf_overhead_s_{long_steps}.csv`](llama_hf_overhead_s_{long_steps}.csv) (median of {repeats} runs).
"""
    else:
        long_section = ""
        compare_long = ""

    compare_rows = []
    for size in LADDER:
        c = grp(size, "cuda", "overlap")
        n = grp(size, "nntile", "overlap")
        llama_ratio = g(n, "metrics", "train_wall_s", "mean") / g(
            c, "metrics", "train_wall_s", "mean"
        )
        g2_ratio = GPT2_REF["ratios"].get(size)
        neox_ratio = GPT_NEOX_REF["ratios"].get(size)
        g2_cell = f"{g2_ratio:.2f}×" if g2_ratio is not None else "—"
        neox_cell = f"{neox_ratio:.2f}×" if neox_ratio is not None else "—"
        compare_rows.append(
            f"| {SIZE_LABEL[size]} | {g2_cell} | {neox_cell} | **{llama_ratio:.2f}×** |"
        )
    compare_section = f"""## Comparison to GPT-2 / GPT-NeoX (same ladder geometry)

See [`gpt2_hf_overhead_scale.md`](gpt2_hf_overhead_scale.md) and
[`gpt_neox_hf_overhead_scale.md`](gpt_neox_hf_overhead_scale.md) for the GPT-2 and
GPT-NeoX 10× runs (A40, Aug 2026). All **Llama** configs below use one GPU
(see recipe).

| Size | GPT-2 nntile/CUDA | GPT-NeoX nntile/CUDA | Llama nntile/CUDA |
|------|------------------:|---------------------:|------------------:|
{chr(10).join(compare_rows)}
{compare_long}
"""

    # steady sequential compute iter 2
    steady = {}
    for size in LADDER:
        vals = [
            r["iters"][1]["compute"]
            for r in results
            if r["size"] == size
            and r["device"] == "nntile"
            and r["mode"] == "sequential"
            and len(r["iters"]) > 1
            and r["iters"][1].get("compute") is not None
        ]
        steady[size] = statistics.mean(vals)

    return f"""# Llama HF: graph overhead vs width / seqlen

{prelim_block}Ten-step stock HuggingFace **LlamaForCausalLM** on **CUDA** vs **`device=nntile`**.
Depth is **12 layers** for XS–L (MHA: ``num_key_value_heads = num_attention_heads``).
**XL** uses **6 layers** with wider ``hidden_size`` (~same parameter count / VRAM as L).
Width and sequence length grow together with **`seq_len = hidden_size / 2`**. XS uses
the 2 GiB Llama width (`hidden_size=1536` from
[`2gb/llama.json`](../../torch_nntile/examples/2gb/llama.json)) with **12 layers**
instead of that file's 18.

> **VRAM warning.** Same as GPT-2: nntile keeps extra graph buffers. Keep CUDA
> well under the card limit on large configs so nntile stays on-device (no
> StarPU CPU↔GPU paging).

Configs: [`torch_nntile/examples/overhead_llama/`](../../torch_nntile/examples/overhead_llama/).  
Script: [`train_llama_hf.py`](../../torch_nntile/examples/train_llama_hf.py).  
Benchmark runner: [`run_llama_overhead_benchmark.py`](../../torch_nntile/tools/run_llama_overhead_benchmark.py).

## Attention backend

Same as GPT-2 / GPT-NeoX: stock HF Llama with **`attn_implementation="sdpa"`**,
MATH backend pinned on both CUDA and nntile. CUDA runs with `--disable-tf32`.

## Train wall

Same recipe as
[`gpt2_hf_overhead_scale.md`](gpt2_hf_overhead_scale.md): nntile
`record → compile → wait(prev) → run`, wall from first record through final
`wait()`; CUDA synced per iter. Prefetch outside the wall. Iter 1 nntile
`wait=0`; iter 10 `wait` includes the final join.

## Recipe

| | XS | S | M | L | XL |
|--|--:|--:|--:|--:|--:|
| Config | `llama_xs.json` | `llama_s.json` | `llama_m.json` | `llama_l.json` | `llama_xl.json` |
| `num_hidden_layers` | {LAYERS['xs']} | {LAYERS['s']} | {LAYERS['m']} | {LAYERS['l']} | {LAYERS['xl']} |
| `hidden_size` / `num_attention_heads` | {HIDDEN['xs']} | {HIDDEN['s']} | {HIDDEN['m']} | {HIDDEN['l']} | {HIDDEN['xl']} |
| `--seq-len` (`= hidden_size/2`) | **768** | **1024** | **1536** | **2048** | **2880** |
| Params (FP32) | {PARAMS['xs']} | {PARAMS['s']} | {PARAMS['m']} | {PARAMS['l']} | {PARAMS['xl']} |

B=1, 10 steps, seed 42, `--no-shuffle`, MATH SDPA, CUDA `--disable-tf32`,
nntile `--ncpu 0 --ncuda 1 --restrict-cuda`. NVIDIA A40, **GPU 2** for every
config (`CUDA_VISIBLE_DEVICES=2` via the runner). Separate processes
(`PYTHONNOUSERSITE=1`; never import `torch_nntile` in the CUDA child).
**No checkpoints** (`--no-save-checkpoint`; output dirs deleted after each run).

Rerun: 2026-08-26, **{repeats} repeats per configuration** (mean ± stdev;
`{logdir}`). Includes **S nntile {long_steps}-step** steady-state run per repeat.

## Overall (10-step train wall)

| Setup | CUDA wall | nntile wall | nntile/CUDA | record(nntile) | record(torch) | compile | run | wait | host/wall | CUDA loss | nntile loss |
|-------|----------:|------------:|------------:|---------------:|--------------:|--------:|----:|-----:|----------:|----------:|------------:|
{chr(10).join(overall_rows)}

Host = `record(nntile)+record(torch)+compile` (~0.50–0.59 s for 10 steps,
**flat**). Host **share** drops **{' → '.join(host_shares)}**
as GPU work grows.

{loss_section}

Isolated GPU `run+wait` vs CUDA isolated wall:
{', '.join(
    f"{SIZE_LABEL[size]} {ms(g(grp(size, 'nntile', 'overlap')['isolated'], 'run_wait'))} vs "
    f"{ms(g(grp(size, 'cuda', 'overlap')['isolated'], 'cuda_wall'))} s"
    for size in LADDER
)}.

{long_section}
{compare_section}
## Per iteration (mean ± stdev over {repeats} runs)

### XS (`hidden_size=1536`, `T=768`)

| Iter | CUDA wall | record(nntile) | record(torch) | compile | run | wait |
|-----:|----------:|---------------:|--------------:|--------:|----:|-----:|
{iter_mean_table(results, 'xs', 'nntile', 'overlap')}

### S (`hidden_size=2048`, `T=1024`)

| Iter | CUDA wall | record(nntile) | record(torch) | compile | run | wait |
|-----:|----------:|---------------:|--------------:|--------:|----:|-----:|
{iter_mean_table(results, 's', 'nntile', 'overlap')}

### M (`hidden_size=3072`, `T=1536`)

| Iter | CUDA wall | record(nntile) | record(torch) | compile | run | wait |
|-----:|----------:|---------------:|--------------:|--------:|----:|-----:|
{iter_mean_table(results, 'm', 'nntile', 'overlap')}

### L (`hidden_size=4096`, `T=2048`)

| Iter | CUDA wall | record(nntile) | record(torch) | compile | run | wait |
|-----:|----------:|---------------:|--------------:|--------:|----:|-----:|
{iter_mean_table(results, 'l', 'nntile', 'overlap')}

### XL (`hidden_size=5760`, `T=2880`, 6 layers, head_dim=128)

| Iter | CUDA wall | record(nntile) | record(torch) | compile | run | wait |
|-----:|----------:|---------------:|--------------:|--------:|----:|-----:|
{iter_mean_table(results, 'xl', 'nntile', 'overlap')}

## Isolated extra step (mean ± stdev over {repeats} runs)

| Setup | record(nntile) | record(torch) | compile | run | wait | run+wait | CUDA isolated |
|-------|---------------:|--------------:|--------:|----:|-----:|---------:|--------------:|
{chr(10).join(iso_rows)}

| Setup | Full isolated (record+compile+run+wait) | Hidden host (`run+wait`) | Saved |
|-------|----------------------------------------:|-------------------------:|------:|
{chr(10).join(hidden_rows)}

## Sequential prep vs compute (`--wait-after-run`)

| Setup | CUDA wall | sequential wall | prep | compute | compute/CUDA | prep/wall |
|-------|----------:|----------------:|-----:|--------:|-------------:|----------:|
{chr(10).join(seq_rows)}

Sequential nntile loss: {', '.join(seq_loss)}.

### Per iteration (prep / compute, mean ± stdev)

#### XS (`T=768`)

| Iter | prep | compute | record(nntile) | record(torch) | compile | run | wait |
|-----:|-----:|--------:|---------------:|--------------:|--------:|----:|-----:|
{iter_mean_table(results, 'xs', 'nntile', 'sequential')}

#### S (`T=1024`)

| Iter | prep | compute | record(nntile) | record(torch) | compile | run | wait |
|-----:|-----:|--------:|---------------:|--------------:|--------:|----:|-----:|
{iter_mean_table(results, 's', 'nntile', 'sequential')}

#### M (`T=1536`)

| Iter | prep | compute | record(nntile) | record(torch) | compile | run | wait |
|-----:|-----:|--------:|---------------:|--------------:|--------:|----:|-----:|
{iter_mean_table(results, 'm', 'nntile', 'sequential')}

#### L (`T=2048`)

| Iter | prep | compute | record(nntile) | record(torch) | compile | run | wait |
|-----:|-----:|--------:|---------------:|--------------:|--------:|----:|-----:|
{iter_mean_table(results, 'l', 'nntile', 'sequential')}

Steady compute after iter 1 (mean over repeats): ~{steady['xs']:.3f} s (XS),
~{steady['s']:.3f} s (S), ~{steady['m']:.3f} s (M), ~{steady['l']:.3f} s (L).

## Takeaways

1. **`seq_len = hidden_size / 2`**, 12 layers, MATH SDPA attention.
2. **Graph host overhead is flat** (~0.5 s / 10 steps); share falls as GPU
   work grows ({host_shares[0]} → {host_shares[3]}).
3. **With VRAM headroom, nntile matches or beats CUDA on wall time**
   ({', '.join(ratios)}).
4. **Sequential GPU time** (`run+wait`): **{' → '.join(seq_compute_ratios)}** CUDA.
5. Timings are **mean ± stdev over {repeats} runs** on the same GPU.
6. Check **CUDA vs nntile loss** above for training parity beyond XS.
7. **{long_steps}-step S** wall **{ms_s(g(s1k, 'metrics', 'train_wall_s')) if s1k else 'n/a'}** — see section above.

## How to reproduce

```bash
export TORCH_LIB_DIR="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export NNTILE_BUILD_DIR=$PWD/build TORCH_NNTILE_BUILD_DIR=$PWD/build
export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{TORCH_LIB_DIR}}:$PWD/build/nntile:$PWD/build/torch_nntile:/opt/starpu/lib"
export STARPU_SILENT=1 STARPU_FXT_TRACE=0 STARPU_WORKERS_NOBIND=1

python3 torch_nntile/tools/run_llama_overhead_benchmark.py \\
  --logdir /tmp/llama_overhead_x10_YYYYMMDD --gpu 2 --repeats 10 --long-steps 100

python3 torch_nntile/tools/update_llama_overhead_doc.py \\
  --summary /tmp/llama_overhead_x10_YYYYMMDD/results_summary.json \\
  --results /tmp/llama_overhead_x10_YYYYMMDD/results.json
```
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument(
        "--preliminary-note",
        type=str,
        default="",
        help="Banner note (e.g. '3/10 repeats, GPU 0')",
    )
    parser.add_argument("--output", type=Path, default=DOC)
    parser.add_argument(
        "--merge-summary",
        type=Path,
        action="append",
        default=[],
        help="Additional results_summary.json files to merge (e.g. XL-only run)",
    )
    parser.add_argument(
        "--merge-results",
        type=Path,
        action="append",
        default=[],
        help="Additional results.json files to merge",
    )
    args = parser.parse_args()
    summary = load_summary(args.summary)
    for extra_path in args.merge_summary:
        summary = merge_summaries(summary, load_summary(extra_path))
    results = json.loads(args.results.read_text(encoding="utf-8"))
    for extra_path in args.merge_results:
        results.extend(json.loads(extra_path.read_text(encoding="utf-8")))
    logdir = args.logdir or str(args.summary.parent)
    long_steps = int(summary.get("long_steps", 100))
    long_mode = f"{long_steps}step"
    text = render_doc(summary, results, logdir, args.preliminary_note)
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote {args.output}")

    csv_path = REPO / "docs" / "dev" / f"llama_hf_overhead_s_{long_steps}.csv"
    svg_path = REPO / "docs" / "dev" / f"llama_hf_overhead_s_{long_steps}.svg"
    if write_long_plots(
        results,
        long_mode=long_mode,
        csv_path=csv_path,
        svg_path=svg_path,
        title=f"Llama S nntile host overhead per iteration ({long_steps} steps)",
    ):
        print(f"wrote {csv_path}")
        print(f"wrote {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
