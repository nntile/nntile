#!/usr/bin/env python3
"""Insert -1 (no worker pin) as first arg to core::op and starpu::submit test calls."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "nntile/tests"
CORE_INC = ROOT / "nntile/include/nntile/core"

CORE_OPS = sorted(
    {
        p.stem
        for p in CORE_INC.glob("*.hh")
        if p.stem not in ("tile", "execution_worker", "execution_schedule", "tile")
    },
    key=len,
    reverse=True,
)


def patch_core_calls(text: str) -> str:
    text = re.sub(
        r"nntile::core::(\w+)<([^>]+)>\(\s*(?!-1)",
        r"nntile::core::\1<\2>(-1, ",
        text,
    )
    for op in CORE_OPS:
        text = re.sub(
            rf"\b{re.escape(op)}<([^>]+)>\(\s*(?!-1)",
            rf"{op}<\1>(-1, ",
            text,
        )
        text = re.sub(
            rf"\b{re.escape(op)}_async<([^>]+)>\(\s*(?!-1)",
            rf"{op}_async<\1>(-1, ",
            text,
        )
        text = re.sub(
            rf"\b{re.escape(op)}\(\s*(?!-1)",
            rf"{op}(-1, ",
            text,
        )
    return text


def patch_starpu_submit(text: str) -> str:
    text = re.sub(
        r"\.submit<std::tuple<([^>]+)>>\(\s*(?!-1)",
        r".submit<std::tuple<\1>>(-1, ",
        text,
    )
    text = re.sub(
        r"(\b\w+)\.submit\(\s*(?!-1)",
        r"\1.submit(-1, ",
        text,
    )
    return text


def main() -> None:
    for path in TESTS.rglob("*.cc"):
        original = path.read_text()
        updated = patch_starpu_submit(patch_core_calls(original))
        if updated != original:
            path.write_text(updated)
            print(f"patched {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
