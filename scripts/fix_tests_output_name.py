#!/usr/bin/env python3
"""Remove legacy output_name args from tests; use set_name when needed."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"


def fix_text(text: str) -> str:
    text = text.replace(
        "gelu(static_cast<NNGraph::TensorNode*>(nullptr), \"out\")",
        "gelu(static_cast<NNGraph::TensorNode*>(nullptr))",
    )

    # gemm(..., "name", alpha -> gemm(..., alpha
    for pref in (r"gt::gemm", r"gemm"):
        text = re.sub(
            rf"({pref}\(\s*[^,]+,\s*[^,]+),\s*\"[^\"]*\"\s*,\s*",
            r"\1, ",
            text,
        )

    # multiply(x, y, "z", alpha)
    for pref in (r"gt::multiply", r"\bmultiply"):
        text = re.sub(
            rf"({pref}\(\s*[^,]+,\s*[^,]+),\s*\"[^\"]*\"\s*,\s*",
            r"\1, ",
            text,
        )

    # concat(a, b, axis, "name")
    for pref in (r"gt::concat", r"\bconcat"):
        text = re.sub(
            rf"({pref}\(\s*[^,]+,\s*[^,]+,\s*[^,]+),\s*\"[^\"]*\"\s*\)",
            r"\1)",
            text,
        )

    # add_slice(..., "out", axis)
    for pref in (r"gt::add_slice", r"\badd_slice"):
        text = re.sub(
            rf"({pref}\(\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+),\s*"
            r'"[^\"]*"\s*,\s*',
            r"\1, ",
            text,
        )

    # add_fiber: drop string before axis (tensor graph)
    text = re.sub(
        r"(\bgt::add_fiber\(\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+),\s*"
        r'"[^\"]*"\s*,\s*',
        r"\1, ",
        text,
    )

    # NN add_fiber multiline: ..., "out",\n -> ...,
    text = re.sub(
        r"add_fiber\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*,\s*"
        r'"[^"]*"\s*,\s*\n\s*',
        r"add_fiber(\1, \2, \3, \4, ",
        text,
    )

    # NN add — auto* and simple assignments only (not Catch macros)
    text = re.sub(
        r"(\bauto\*\s+\w+\s*=\s*)add\(\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*"
        r"([^,]+),\s*"
        r'"([^"]*)"\s*\)',
        r'\1add(\2, \3, \4, \5)->set_name("\6")',
        text,
    )
    text = re.sub(
        r"(\s+)([a-zA-Z_]\w*)\s*=\s*add\(\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*"
        r"([^,]+),\s*"
        r'"([^"]*)"\s*\)\s*;',
        r'\1\2 = add(\3, \4, \5, \6)->set_name("\7");',
        text,
    )

    for ns in ("gt", "tg"):
        text = re.sub(
            rf"(\s*)auto\*\s+(\w+)\s*=\s*{ns}::add\(\s*([^,]+),\s*([^,]+),\s*"
            rf"([^,]+),\s*([^,]+)\s*,\s*\"([^\"]*)\"\s*\)\s*;",
            lambda m, ns=ns: (
                f"{m.group(1)}auto* {m.group(2)} = "
                f"{ns}::add({m.group(3)}, {m.group(4)}, {m.group(5)}, "
                f"{m.group(6)});\n{m.group(1)}{m.group(2)}"
                f'->set_name("{m.group(7)}");'
            ),
            text,
        )

    # gt::softmax(maxsumexp, src, "dst", alpha, axis)
    text = re.sub(
        r"(\bgt::softmax\(\s*[^,]+,\s*[^,]+),\s*\"[^\"]*\"\s*,\s*",
        r"\1, ",
        text,
    )

    # gt::rope(sin, cos, src, "dst")
    text = re.sub(
        r"(\bgt::rope\(\s*[^,]+,\s*[^,]+,\s*[^,]+),\s*\"[^\"]*\"\s*\)",
        r"\1)",
        text,
    )
    unary_gt_ops = ("gelu", "relu", "silu", "gelutanh")
    for op in unary_gt_ops:
        text = re.sub(
            rf"(\s*)auto\*\s+(\w+)\s*=\s*gt::{op}\(\s*([^,]+)\s*,\s*"
            r'"([^"]*)"\s*\)\s*;',
            lambda m, op=op: (
                f"{m.group(1)}auto* {m.group(2)} = "
                f"gt::{op}({m.group(3)});\n{m.group(1)}{m.group(2)}"
                f'->set_name("{m.group(4)}");'
            ),
            text,
        )

        text = re.sub(
            rf"(\s*)([^\n=]+?)\s+(\w+)\s*=\s*gt::{op}\(\s*([^,]+)\s*,\s*"
            r'"([^"]*)"\s*\)\s*;',
            lambda m, op=op: (
                f"{m.group(1)}{m.group(2)} {m.group(3)} = "
                f"gt::{op}({m.group(4)});\n{m.group(1)}{m.group(3)}"
                f'->set_name("{m.group(5)}");'
            ),
            text,
        )

    # NN gelu(x, "z") — gelu returns NN TensorNode*
    text = re.sub(
        r"\bgelu\(\s*([^,]+)\s*,\s*\"([^\"]*)\"\s*\)",
        r'gelu(\1)->set_name("\2")',
        text,
    )

    # gt::transpose(alpha, src, "dst", ndim)
    text = re.sub(
        r"(\s*)auto\*\s+(\w+)\s*=\s*gt::transpose\(\s*([^,]+),\s*([^,]+),\s*"
        r'"([^"]*)"\s*,\s*([^)]+)\)\s*;',
        lambda m: (
            f"{m.group(1)}auto* {m.group(2)} = "
            f"gt::transpose({m.group(3)}, {m.group(4)}, {m.group(6)});\n"
            f'{m.group(1)}{m.group(2)}->set_name("{m.group(5)}");'
        ),
        text,
    )

    # Any remaining 5-arg gt::add / tg::add (e.g. REQUIRE_THROWS)
    for ns in ("gt", "tg"):
        text = re.sub(
            rf"\b{ns}::add\(\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+)\s*"
            r',\s*"[^"]*"\s*\)',
            rf"{ns}::add(\1, \2, \3, \4)",
            text,
        )

    return text


def main() -> int:
    paths = sorted(TESTS.rglob("*.cc"))
    changed = []
    for path in paths:
        raw = path.read_text()
        new = fix_text(raw)
        if new != raw:
            path.write_text(new)
            changed.append(path)
    for p in changed:
        print(p.relative_to(ROOT))
    print(f"updated {len(changed)} files", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
