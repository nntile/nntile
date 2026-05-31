#!/usr/bin/env python3
"""Add explicit starpu_worker_hint to StarPU submit / core / tile call chain."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STARPU_INC = ROOT / "nntile/include/nntile/starpu"
STARPU_SRC = ROOT / "nntile/src/starpu"
CORE_INC = ROOT / "nntile/include/nntile/core"
CORE_SRC = ROOT / "nntile/src/core"
TILE_OPS = ROOT / "nntile/src/tile/ops"
SKIP_STARPU = {"codelet.hh", "config.hh", "handle.hh", "task_insert.hh"}


def patch_starpu_headers() -> None:
    for path in STARPU_INC.glob("*.hh"):
        if path.name in SKIP_STARPU:
            continue
        text = path.read_text()
        if "void submit(" not in text or "int starpu_worker_hint" in text:
            continue
        text = re.sub(
            r"void submit\(\s*\n",
            "void submit(\n        int starpu_worker_hint,\n",
            text,
            count=1,
        )
        path.write_text(text)


def patch_starpu_sources() -> None:
    for path in STARPU_SRC.glob("*.cc"):
        text = path.read_text()
        if "nntile_starpu_task_insert" not in text:
            continue
        if "int starpu_worker_hint" not in text:
            text = re.sub(
                r"::submit\(\s*\n(\s+)(?!int starpu_worker_hint)",
                r"::submit(\n\1int starpu_worker_hint,\n\1",
                text,
                count=1,
            )
        text = text.replace(
            "nntile_starpu_task_insert(&codelet,",
            "nntile_starpu_task_insert(&codelet, starpu_worker_hint,",
        )
        text = re.sub(
            r"\.submit<std::tuple<([^>]+)>>\(\s*\n(\s+)(?!starpu_worker_hint)",
            r".submit<std::tuple<\1>>(\n\2starpu_worker_hint,\n\2",
            text,
        )
        text = re.sub(
            r"\.submit<std::tuple<([^>]+)>>\((?!starpu_worker_hint)",
            r".submit<std::tuple<\1>>(starpu_worker_hint, ",
            text,
        )
        path.write_text(text)


def patch_core_header(path: Path) -> None:
    text = path.read_text()
    if "int starpu_worker_hint" in text:
        return
    for suffix in ("_async", ""):
        text = re.sub(
            rf"void (\w+){suffix}\(\s*\n(\s+)(?!int starpu_worker_hint)",
            rf"void \1{suffix}(\n\2int starpu_worker_hint,\n\2",
            text,
            count=1,
        )
    path.write_text(text)


def patch_core_source(path: Path) -> None:
    text = path.read_text()
    if "starpu::" not in text:
        return
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # template<typename T> followed by void foo_async( or void foo(
        if (
            line.strip() == "template<typename T>"
            and i + 1 < len(lines)
            and re.match(r"void \w+(_async)?\(", lines[i + 1])
            and "starpu_worker_hint" not in lines[i + 1]
        ):
            out.append(line)
            m = re.match(r"(void \w+(?:_async)?)\(\s*$", lines[i + 1])
            if m:
                out.append(f"{m.group(1)}(\n")
                out.append("        int starpu_worker_hint,\n")
                i += 2
                continue
        out.append(line)
        i += 1
    text = "".join(out)
    text = re.sub(
        r"starpu::(\w+)\.submit<std::tuple<([^>]+)>>\(\s*\n(\s+)(?!starpu_worker_hint)",
        r"starpu::\1.submit<std::tuple<\2>>(\n\3starpu_worker_hint,\n\3",
        text,
    )
    text = re.sub(
        r"starpu::(\w+)\.submit<std::tuple<([^>]+)>>\((?!starpu_worker_hint)",
        r"starpu::\1.submit<std::tuple<\2>>(starpu_worker_hint, ",
        text,
    )
    path.write_text(text)


def patch_tile_ops() -> None:
    for path in TILE_OPS.glob("*.cc"):
        text = path.read_text()
        if "nntile::core::" not in text or "starpu_worker_hint()" in text:
            continue
        hint = "runtime.starpu_worker_hint()"
        if re.search(r"void run_\w+\([^)]*\bRuntime\s*&\s*rt\b", text):
            hint = "rt.starpu_worker_hint()"
        text = re.sub(
            r"nntile::core::(\w+)<T>\(",
            f"nntile::core::\\1<T>({hint}, ",
            text,
        )
        path.write_text(text)


def patch_tests_core() -> None:
    tests = ROOT / "nntile/tests/core"
    for path in tests.glob("*.cc"):
        text = path.read_text()
        if ".submit<std::tuple" not in text:
            continue
        text = re.sub(
            r"\.submit<std::tuple<([^>]+)>>\(\s*\n(\s+)(?!-1)(?!starpu_worker_hint)",
            r".submit<std::tuple<\1>>(\n\2-1,\n\2",
            text,
        )
        text = re.sub(
            r"\.submit<std::tuple<([^>]+)>>\((?!-1)(?!starpu_worker_hint)",
            r".submit<std::tuple<\1>>(-1, ",
            text,
        )
        path.write_text(text)


def main() -> None:
    patch_starpu_headers()
    patch_starpu_sources()
    for path in CORE_INC.glob("*.hh"):
        if path.name in ("tile.hh", "execution_schedule.hh", "execution_worker.hh"):
            continue
        patch_core_header(path)
    for path in CORE_SRC.glob("*.cc"):
        if path.name == "execution_schedule.cc":
            continue
        patch_core_source(path)
    patch_tile_ops()
    patch_tests_core()
    print("migration done")


if __name__ == "__main__":
    main()
