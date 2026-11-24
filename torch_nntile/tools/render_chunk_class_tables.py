#!/usr/bin/env python3
"""Render the tables of docs/dev/gemm_routes_chunk_class.md.

The input is the CSV output of ``bench_gemm_chunked_gpt2`` (its ``ENGINE``,
``INFO`` and ``SAMPLES`` lines), one file per engine build: the staged
routes exist only at a build's own chunk class, so a file carries the
staged, chunked and cuBLAS routes of exactly that class. The defaults are
the files committed next to the doc, for the 1, 2 and 4 MiB classes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import render_gemm_chunked_gpt2_tables as routes

STEM = "gemm_routes_chunk_class"
DEFAULTS = tuple(routes.DOCS / f"{STEM}_{cls}mib.csv" for cls in (1, 2, 4))
STAGES = ("staged", "chunked")


def per_step_table(engines: list[routes.Engine]) -> list[str]:
    """GEMM time per training step by route and chunk class. The chunked
    total is the native route where the family runs the row and staged
    elsewhere."""
    head = "| model | tokens/step | cuBLAS ms | " + " | ".join(
        f"{name} class: staged · chunked" for name, _, _, _ in engines
    )
    out = [head, "|---|---:|---:|" + "---|---" * len(engines)]
    for model in routes.MODELS:
        for tokens in routes.TOKENS:
            stats = [
                routes.step_time(rows, samples, model, tokens, True)
                for _, _, rows, samples in engines
            ]
            if stats[0]["cublas"] == 0:
                continue
            cublas = sum(s["cublas"] for s in stats) / len(stats)
            cells = []
            for s in stats:
                staged = routes.ratio(s["staged"], s["cublas"])
                native = routes.ratio(s["native"], s["cublas"])
                cells.append(
                    f"{s['staged'] / 1e3:.1f} ({staged})"
                    f" · {s['native'] / 1e3:.1f} ({native})"
                )
            spec = routes.MODEL_SPEC[model]
            out.append(
                f"| {model} ({spec}) | {tokens} | {cublas / 1e3:.1f} | "
                + " | ".join(cells)
                + " |"
            )
    return out


def row_tables(engines: list[routes.Engine], tokens: int) -> list[str]:
    """Per-row tables: cuBLAS microseconds, then staged and chunked per
    chunk class as microseconds and as ratios of cuBLAS."""
    out: list[str] = []
    names = [name for name, _, _, _ in engines]
    for model in routes.MODELS:
        keys = routes.keys_of(engines[0][2], model, tokens)
        if not keys:
            continue
        spec = routes.MODEL_SPEC[model]
        out += ["", f"**GPT-2 {model} ({spec}), {tokens} tokens/step**", ""]
        out.append(
            "| GEMM row | m, n, k × batch | calls | cuBLAS µs | "
            + " | ".join(
                f"{stage} {name}" for name in names for stage in STAGES
            )
            + " |"
        )
        out.append("|---|---|---:|---:|" + "---|---" * len(engines))
        tot = {(name, stage): 0.0 for name in names for stage in STAGES}
        step_cublas = 0.0
        no_run: dict[str, list[str]] = {name: [] for name in names}
        for key in keys:
            row = engines[0][2][key]
            cublas = sum(
                samples[key]["cublas"] for _, _, _, samples in engines
            )
            cublas /= len(engines)
            calls = row["calls"]
            step_cublas += calls * cublas
            cells = []
            for name, _, _, samples in engines:
                s = samples.get(key, {})
                for stage in STAGES:
                    value = s.get(stage)
                    missing = value is None
                    # Without a chunked route the step total keeps what
                    # the wrapper would do instead (staged).
                    if missing:
                        value = s.get("staged", cublas)
                    tot[name, stage] += calls * value
                    if missing:
                        cells.append("n/a")
                        if stage == "chunked":
                            no_run[name].append(f"`{key[2]}`")
                    else:
                        cells.append(
                            f"{value:.0f} ({routes.ratio(value, cublas)})"
                        )
            shape = f"{row['m']}, {row['n']}, {row['k']} × {row['batch']}"
            out.append(
                f"| `{key[2]}` | {shape} | {calls} | {cublas:.0f} | "
                + " | ".join(cells)
                + " |"
            )
        cells = []
        for name in names:
            for stage in STAGES:
                total = tot[name, stage]
                rel = routes.ratio(total, step_cublas)
                cells.append(f"**{total / 1e3:.1f} ({rel})**")
        step = f"**{step_cublas / 1e3:.1f}**"
        out.append(
            f"| **per training step (ms)** | | | {step} | "
            + " | ".join(cells)
            + " |"
        )
        for name in names:
            if no_run[name]:
                listed = ", ".join(no_run[name])
                out += [f"No chunked route at {name}: {listed}."]
    return out


def operand_table(engines: list[routes.Engine]) -> list[str]:
    """Chunks per MLP operand at each class, read off the ``mlp_fc_fwd``
    rows: A is the hidden state S x H, B the fc weight 4H x H and C the
    fc activation S x 4H. One cell reads 1 / 2 / 4 MiB class."""
    out = [
        "| model | tokens | hidden state S×H | fc weight 4H×H | "
        "fc activation S×4H |",
        "|---|---:|---|---|---|",
    ]
    for model in routes.MODELS:
        for tokens in routes.TOKENS:
            key = (model, tokens, "mlp_fc_fwd")
            cells = []
            for col in range(3):
                counts = [
                    rows[key]["chunks"].split("/")[col]
                    for _, _, rows, _ in engines
                ]
                cells.append(" / ".join(counts))
            out.append(f"| {model} | {tokens} | " + " | ".join(cells) + " |")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        nargs="*",
        type=Path,
        help="benchmark CSV per engine build (default: the committed ones)",
    )
    args = parser.parse_args()
    engines: list[routes.Engine] = []
    for path in args.csv or DEFAULTS:
        engines.extend(routes.load(path))
    print("## GEMM time per training step\n")
    print(*per_step_table(engines), sep="\n")
    names = " / ".join(name for name, _, _, _ in engines)
    print(f"\n## Chunks per MLP operand, {names} class\n")
    print(*operand_table(engines), sep="\n")
    for tokens in routes.TOKENS:
        print(f"\n## Per row, {tokens} tokens/step\n")
        print(*row_tables(engines, tokens), sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
