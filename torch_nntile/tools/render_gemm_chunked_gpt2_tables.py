#!/usr/bin/env python3
"""Render the tables of docs/dev/gemm_chunked_gpt2_routes.md.

The input is the CSV output of ``bench_gemm_chunked_gpt2`` (its ``ENGINE``,
``INFO`` and ``SAMPLES`` lines). The defaults are the files committed next to
the doc: one per engine build (the staged body exists only at the engine's
chunk class) and one for the classes at which only the native family runs.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs" / "dev"
STEM = "gemm_chunked_gpt2_routes"
DEFAULTS = (
    DOCS / f"{STEM}_4mib.csv",
    DOCS / f"{STEM}_16mib.csv",
)
DEFAULT_NATIVE_ONLY = DOCS / f"{STEM}_1-2-8mib.csv"

MODELS = ("small", "medium", "large", "xl")
MODEL_SPEC = {
    "small": "H=768, 12 heads, 12 layers",
    "medium": "H=1024, 16 heads, 24 layers",
    "large": "H=1280, 20 heads, 36 layers",
    "xl": "H=1600, 25 heads, 48 layers",
}
TOKENS = (1024, 4096)
# The two routes shown per engine build in the per-row tables.
STAGES = ("staged", "native")

# One INFO record.
Row = dict[str, Any]
# One GEMM row inside one engine build: (model, tokens per step, GEMM row).
Key = tuple[str, int, str]
Rows = dict[Key, Row]
# One record's timed routes: route name -> median microseconds.
Samples = dict[Key, dict[str, float]]
# One engine build: its label and chunk class in MiB.
Engine = tuple[str, int, Rows, Samples]


def load(path: Path) -> list[Engine]:
    """Read one benchmark file into one record per chunk class in it."""
    rows: dict[int, Rows] = collections.defaultdict(dict)
    samples: dict[int, Samples] = collections.defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        f = line.split(",")
        if len(f) < 11 or f[0] not in ("INFO", "SAMPLES"):
            continue
        cls = int(f[3])
        key = (f[1], int(f[2]), f[4])
        if f[0] == "SAMPLES":
            samples[cls].setdefault(key, {})[f[5]] = float(f[7])
        elif len(f) >= 21:
            rows[cls][key] = {
                "m": int(f[6]),
                "n": int(f[7]),
                "k": int(f[8]),
                "batch": int(f[9]),
                "calls": int(f[10]),
                "chunks": f[11],
                "layout": f[12],
                "supported": f[13] == "y",
                "owned": f[14] == "y",
                "engine_leg": f[16] == "y",
            }
    return [
        (f"{cls} MiB", cls, rows[cls], samples[cls])
        for cls in sorted(rows)
        if rows[cls]
    ]


def ratio(value: float, base: float) -> str:
    return f"{value / base:.2f}×"


def keys_of(rows: Rows, model: str, tokens: int) -> list[Key]:
    """The model's rows at one token count, in the benchmark's order."""
    keys = [k for k in rows if k[0] == model and k[1] == tokens]
    return sorted(keys, key=lambda k: (-rows[k]["calls"], k[2]))


def step_time(
    rows: Rows,
    samples: Samples,
    model: str,
    tokens: int,
    production: bool,
) -> dict[str, float]:
    """GEMM time per training step (ms) by route, plus row counts.

    ``native`` is the native family's best configuration where it can run
    the row and staged (plain cuBLAS where no staged route exists)
    elsewhere. ``production`` is the wrapper's flag-off routing: plain
    cuBLAS when every operand is one chunk, else the owned native
    configuration, else staged (the CUTLASS leg where the winners table has
    the row). It needs the staged routes, so it is off for the classes
    where only the native family runs.
    """
    tot: dict[str, float] = collections.defaultdict(float)
    for key in keys_of(rows, model, tokens):
        row, s, calls = rows[key], samples[key], rows[key]["calls"]
        cublas = s["cublas"]
        staged = s.get("staged", cublas)
        native = s.get("chunked")
        tot["cublas"] += calls * cublas
        tot["native"] += calls * (staged if native is None else native)
        tot["launched"] += native is not None
        if not production:
            continue
        tot["staged"] += calls * staged
        if row["layout"] == "fff":
            prod = cublas
        elif row["owned"]:
            prod = s["chunked_prod"]
            tot["owned"] += 1
        elif row["engine_leg"] and "staged_cut" in s:
            prod = s["staged_cut"]
        else:
            prod = staged
        tot["production"] += calls * prod
    return tot


def per_step_table(engines: list[Engine]) -> list[str]:
    head = "| model | tokens/step | cuBLAS ms | " + " | ".join(
        f"{name}: staged · native · production" for name, _, _, _ in engines
    )
    out = [head, "|---|---:|---:|" + "---|---" * len(engines)]
    for model in MODELS:
        for tokens in TOKENS:
            stats = [
                step_time(rows, samples, model, tokens, True)
                for _, _, rows, samples in engines
            ]
            if stats[0]["cublas"] == 0:
                continue
            cublas = sum(s["cublas"] for s in stats) / len(stats)
            cells = []
            for s in stats:
                rel = ratio(s["staged"], s["cublas"])
                cell = f"{s['staged'] / 1e3:.1f} ({rel})"
                cell += (
                    f" · {s['native'] / 1e3:.1f} "
                    f"({ratio(s['native'], s['cublas'])})"
                )
                cell += (
                    f" · {s['production'] / 1e3:.1f} "
                    f"({ratio(s['production'], s['cublas'])})"
                )
                cells.append(cell)
            spec = MODEL_SPEC[model]
            out.append(
                f"| {model} ({spec}) | {tokens} | {cublas / 1e3:.1f} | "
                + " | ".join(cells)
                + " |"
            )
    return out


def native_only_table(engines: list[Engine]) -> list[str]:
    """The native family alone at the classes without a staged route."""
    head = "| model | tokens/step | cuBLAS ms | " + " | ".join(
        f"{name} class: native, rows launched" for name, _, _, _ in engines
    )
    rule = "|---|---:|---:|" + "---|---" * len(engines)
    out = [head, rule]
    for model in MODELS:
        for tokens in TOKENS:
            stats = [
                step_time(rows, samples, model, tokens, False)
                for _, _, rows, samples in engines
            ]
            if stats[0]["cublas"] == 0:
                continue
            cells = []
            for s in stats:
                rel = ratio(s["native"], s["cublas"])
                cells.append(
                    f"{s['native'] / 1e3:.1f} ({rel}), {int(s['launched'])}/15"
                )
            cublas = stats[0]["cublas"] / 1e3
            out.append(
                f"| {model} | {tokens} | {cublas:.1f} | "
                + " | ".join(cells)
                + " |"
            )
    return out


def row_tables(engines: list[Engine], tokens: int) -> list[str]:
    """Per-row tables: cuBLAS microseconds, then the routes as ratios."""
    out: list[str] = []
    names = [name for name, _, _, _ in engines]
    for model in MODELS:
        keys = keys_of(engines[0][2], model, tokens)
        if not keys:
            continue
        spec = MODEL_SPEC[model]
        out += ["", f"**GPT-2 {model} ({spec}), {tokens} tokens/step**", ""]
        out.append(
            "| GEMM row | m, n, k × batch | calls | cuBLAS µs | "
            + " | ".join(
                f"{stage} {name}" for name in names for stage in STAGES
            )
            + " |"
        )
        out.append("|---|---|---:|---:|" + "---|---" * (2 * len(engines)))
        tot: dict[str, float] = collections.defaultdict(float)
        owned: dict[str, list[str]] = {name: [] for name in names}
        for key in keys:
            row = engines[0][2][key]
            cublas = sum(
                samples[key]["cublas"] for _, _, _, samples in engines
            )
            cublas /= len(engines)
            calls = row["calls"]
            tot["cublas"] += calls * cublas
            cells = []
            for name, _, rows, samples in engines:
                other = rows.get(key)
                s = samples.get(key, {})
                for stage in STAGES:
                    value = s.get(stage if stage != "native" else "chunked")
                    if other is None or value is None:
                        # No native run for the row: the step total keeps
                        # what the wrapper would do instead (staged).
                        cells.append("n/a")
                        value = s.get("staged", cublas)
                    else:
                        cells.append(f"{value:.0f} ({ratio(value, cublas)})")
                    tot[name + stage] += calls * value
                # An all-1-chunk call never reaches the native family:
                # the wrapper's plain-cuBLAS dispatch returns first.
                if (
                    other is not None
                    and other["owned"]
                    and other["layout"] != "fff"
                ):
                    owned[name].append(f"`{key[2]}`")
            shape = f"{row['m']}, {row['n']}, {row['k']} × {row['batch']}"
            out.append(
                f"| `{key[2]}` | {shape} | {calls} | {cublas:.0f} | "
                + " | ".join(cells)
                + " |"
            )
        cells = []
        for name in names:
            for stage in STAGES:
                value = tot[name + stage]
                rel = ratio(value, tot["cublas"])
                cells.append(f"**{value / 1e3:.1f} ({rel})**")
        step_ms = tot["cublas"] / 1e3
        out.append(
            f"| **per training step (ms)** | | | **{step_ms:.1f}** | "
            + " | ".join(cells)
            + " |"
        )
        for name in names:
            listed = ", ".join(owned[name]) if owned[name] else "none"
            out += ["", f"Production routes natively at {name}: {listed}."]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        nargs="*",
        type=Path,
        help="benchmark CSV per engine build (default: the committed ones)",
    )
    parser.add_argument(
        "--native-only",
        type=Path,
        default=DEFAULT_NATIVE_ONLY,
        help="benchmark CSV for the classes without a staged route",
    )
    args = parser.parse_args()
    engines: list[Engine] = []
    for path in args.csv or DEFAULTS:
        engines.extend(load(path))
    print("## GEMM time per training step\n")
    print(*per_step_table(engines), sep="\n")
    print("\n## Native family alone at the other classes\n")
    print(*native_only_table(load(args.native_only)), sep="\n")
    for tokens in TOKENS:
        print(f"\n## Per row, {tokens} tokens/step\n")
        print(*row_tables(engines, tokens), sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
