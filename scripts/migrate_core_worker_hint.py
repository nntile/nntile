#!/usr/bin/env python3
"""Add int starpu_worker_hint to core tile API."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_INC = ROOT / "nntile/include/nntile/core"
CORE_SRC = ROOT / "nntile/src/core"
SKIP_INC = {"tile.hh", "execution_schedule.hh", "execution_worker.hh"}


def patch_void_decls(text: str) -> str:
    if "starpu_worker_hint" in text:
        return text
    text = re.sub(
        r"void (\w+_async)\(",
        lambda m: f"void {m.group(1)}(int starpu_worker_hint, ",
        text,
    )
    text = re.sub(
        r"void (\w+)\(",
        lambda m: (
            f"void {m.group(1)}(int starpu_worker_hint, "
            if not m.group(1).endswith("_async")
            and "starpu_worker_hint" not in m.group(0)
            else m.group(0)
        ),
        text,
    )
    return text


def patch_source(path: Path) -> None:
    text = path.read_text()
    if "starpu::" not in text:
        return
    text = patch_void_decls(text)
    ops = sorted(set(re.findall(r"starpu::(\w+)\.submit", text)))
    for op in ops:
        text = re.sub(
            rf"starpu::{op}\.submit<std::tuple<([^>]+)>>\(\s*\n(\s+)(?!starpu_worker_hint)",
            rf"starpu::{op}.submit<std::tuple<\1>>(\n\2starpu_worker_hint,\n\2",
            text,
        )
        text = re.sub(
            rf"starpu::{op}\.submit<std::tuple<([^>]+)>>\((?!starpu_worker_hint)",
            rf"starpu::{op}.submit<std::tuple<\1>>(starpu_worker_hint, ",
            text,
        )
        text = re.sub(
            rf"{op}_async<T>\((?!starpu_worker_hint)",
            rf"{op}_async<T>(starpu_worker_hint, ",
            text,
        )
    path.write_text(text)


def main() -> None:
    for path in sorted(CORE_INC.glob("*.hh")):
        if path.name in SKIP_INC:
            continue
        path.write_text(patch_void_decls(path.read_text()))
    for path in sorted(CORE_SRC.glob("*.cc")):
        if path.name == "execution_schedule.cc":
            continue
        patch_source(path)
    print("core migration done")


if __name__ == "__main__":
    main()
