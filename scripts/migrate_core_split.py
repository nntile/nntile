#!/usr/bin/env python3
"""One-shot layout + include/namespace rewrites for core/graph split."""
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
            rf"#include \1nntile/{layer}\2",
            text,
        )
    text = re.sub(
        r'#include\s*([<"])nntile/defs\.h([>"])',
        r"#include \1nntile/defs.h\2",
        text,
    )
    return text


def rewrite_namespaces(text: str) -> str:
    # Order: longer graph paths untouched; add core under nntile for layer ns
    for layer in ("kernel", "starpu", "tile", "tensor", "logger"):
        text = re.sub(
            rf"\bnntile::{layer}::",
            rf"nntile::{layer}::",
            text,
        )
        text = re.sub(
            rf"namespace nntile::{layer}\b",
            rf"namespace nntile::{layer}",
            text,
        )
        text = re.sub(
            rf"}} // namespace nntile::{layer}",
            rf"}} // namespace nntile::{layer}",
            text,
        )
        text = re.sub(
            rf"@namespace nntile::{layer}",
            rf"@namespace nntile::{layer}",
            text,
        )
    # Context and top-level nntile members
    text = re.sub(r"\bnntile::Context\b", "nntile::Context", text)
    text = re.sub(
        r"namespace nntile\s*\{",
        "namespace nntile\n{",
        text,
        count=0,
    )
    # Fix double core if any
    text = text.replace("nntile::", "nntile::")
    text = text.replace("namespace nntile", "namespace nntile")
    text = text.replace("nntile::", "nntile::")
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
        ("src/kernel", "nntile/src/kernel"),
        ("src/starpu", "nntile/src/starpu"),
        ("src/tile", "nntile/src/tile"),
        ("src/tensor", "nntile/src/tensor"),
        ("src/logger", "nntile/src/logger"),
        ("src/context.cc", "nntile/src/context.cc"),
    ]
    inc_moves = [
        ("include/nntile/kernel", "include/nntile/kernel"),
        ("include/nntile/starpu", "include/nntile/starpu"),
        ("include/nntile/tile", "include/nntile/tile"),
        ("include/nntile/tensor", "include/nntile/tensor"),
        ("include/nntile/logger", "include/nntile/logger"),
        ("include/nntile/base_types.hh", "include/nntile/base_types.hh"),
        ("include/nntile/constants.hh", "include/nntile/constants.hh"),
        ("include/nntile/context.hh", "include/nntile/context.hh"),
        ("include/nntile/defs.h.in", "include/nntile/defs.h.in"),
        ("include/nntile/kernel.hh", "include/nntile/kernel.hh"),
        ("include/nntile/starpu.hh", "include/nntile/starpu.hh"),
        ("include/nntile/tile.hh", "include/nntile/tile.hh"),
        ("include/nntile/tensor.hh", "include/nntile/tensor.hh"),
        ("include/nntile/logger.hh", "include/nntile/logger.hh"),
    ]
    test_moves = [
        ("tests/kernel", "nntile/tests/eager/kernel"),
        ("tests/starpu", "nntile/tests/eager/starpu"),
        ("tests/tile", "nntile/tests/eager/tile"),
        ("tests/tensor", "nntile/tests/eager/tensor"),
    ]
    for s, d in src_moves + inc_moves + test_moves:
        if (ROOT / s).exists():
            move_path(s, d)

    changed = 0
    for path in ROOT.rglob("*"):
        if (
            path.is_file()
            and "build" not in path.parts
            and ".git" not in path.parts
        ):
            if process_file(path):
                changed += 1
    print(f"Rewrote {changed} files")


if __name__ == "__main__":
    main()
