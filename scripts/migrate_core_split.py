#!/usr/bin/env python3
"""One-shot physical layout + include/namespace rewrites for core/graph split."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CORE_LAYERS = (
    "kernel",
    "starpu",
    "tile",
    "tensor",
    "context",
    "constants",
    "logger",
    "base_types",
)

def rewrite_includes(text: str) -> str:
    # Manual cleaner pass
    for layer in CORE_LAYERS:
        text = re.sub(
            rf'#include\s*([<"])nntile/{re.escape(layer)}(/|\.hh)',
            rf"#include \1nntile/core/{layer}\2",
            text,
        )
    text = re.sub(
        r'#include\s*([<"])nntile/defs\.h([>"])',
        r"#include \1nntile/core/defs.h\2",
        text,
    )
    return text


def rewrite_namespaces(text: str) -> str:
    # Order: longer graph paths untouched; add core under nntile for layer ns
    for layer in ("kernel", "starpu", "tile", "tensor", "logger"):
        text = re.sub(
            rf"\bnntile::{layer}::",
            rf"nntile::core::{layer}::",
            text,
        )
        text = re.sub(
            rf"namespace nntile::{layer}\b",
            rf"namespace nntile::core::{layer}",
            text,
        )
        text = re.sub(
            rf"}} // namespace nntile::{layer}",
            rf"}} // namespace nntile::core::{layer}",
            text,
        )
        text = re.sub(
            rf"@namespace nntile::{layer}",
            rf"@namespace nntile::core::{layer}",
            text,
        )
    # Context and top-level nntile members
    text = re.sub(r"\bnntile::Context\b", "nntile::core::Context", text)
    text = re.sub(
        r"namespace nntile\s*\{",
        "namespace nntile::core\n{",
        text,
        count=0,
    )
    # Fix double core if any
    text = text.replace("nntile::core::", "nntile::core::")
    text = text.replace("namespace nntile::graph", "namespace nntile::graph")
    text = text.replace("nntile::graph::", "nntile::graph::")
    return text


def process_file(path: Path) -> bool:
    if path.suffix not in {
        ".cc",
        ".cu",
        ".hh",
        ".h",
        ".hpp",
        ".cuh",
        ".py",
        ".md",
        ".cmake",
        ".in",
        ".sh",
        ".yml",
        ".yaml",
    } and path.name not in {"CMakeLists.txt", "Dockerfile"}:
        return False
    try:
        original = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False
    updated = rewrite_includes(original)
    updated = rewrite_namespaces(updated)
    if updated != original:
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def move_path(src: str, dst: str) -> None:
    src_p = ROOT / src
    dst_p = ROOT / dst
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_p), str(dst_p))


def main() -> None:
    # Physical moves
    (ROOT / "src" / "core").mkdir(parents=True, exist_ok=True)
    (ROOT / "include" / "nntile" / "core").mkdir(parents=True, exist_ok=True)
    (ROOT / "tests" / "core").mkdir(parents=True, exist_ok=True)

    src_moves = [
        ("src/kernel", "src/core/kernel"),
        ("src/starpu", "src/core/starpu"),
        ("src/tile", "src/core/tile"),
        ("src/tensor", "src/core/tensor"),
        ("src/logger", "src/core/logger"),
        ("src/context.cc", "src/core/context.cc"),
    ]
    inc_moves = [
        ("include/nntile/kernel", "include/nntile/core/kernel"),
        ("include/nntile/starpu", "include/nntile/core/starpu"),
        ("include/nntile/tile", "include/nntile/core/tile"),
        ("include/nntile/tensor", "include/nntile/core/tensor"),
        ("include/nntile/logger", "include/nntile/core/logger"),
        ("include/nntile/base_types.hh", "include/nntile/core/base_types.hh"),
        ("include/nntile/constants.hh", "include/nntile/core/constants.hh"),
        ("include/nntile/context.hh", "include/nntile/core/context.hh"),
        ("include/nntile/defs.h.in", "include/nntile/core/defs.h.in"),
        ("include/nntile/kernel.hh", "include/nntile/core/kernel.hh"),
        ("include/nntile/starpu.hh", "include/nntile/core/starpu.hh"),
        ("include/nntile/tile.hh", "include/nntile/core/tile.hh"),
        ("include/nntile/tensor.hh", "include/nntile/core/tensor.hh"),
        ("include/nntile/logger.hh", "include/nntile/core/logger.hh"),
    ]
    test_moves = [
        ("tests/kernel", "tests/core/kernel"),
        ("tests/starpu", "tests/core/starpu"),
        ("tests/tile", "tests/core/tile"),
        ("tests/tensor", "tests/core/tensor"),
    ]
    for s, d in src_moves + inc_moves + test_moves:
        if (ROOT / s).exists():
            move_path(s, d)

    changed = 0
    for path in ROOT.rglob("*"):
        if path.is_file() and "build" not in path.parts and ".git" not in path.parts:
            if process_file(path):
                changed += 1
    print(f"Rewrote {changed} files")


if __name__ == "__main__":
    main()
