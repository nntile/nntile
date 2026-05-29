#!/usr/bin/env python3
"""Plan B monorepo reorg: git mv layout + include/namespace rewrites."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {
    ".cc",
    ".cu",
    ".hh",
    ".h",
    ".hpp",
    ".cmake",
    ".txt",
    ".md",
    ".py",
    ".pyi",
    ".in",
    ".yml",
    ".yaml",
    ".sh",
    ".rst",
}


def run(cmd: list[str], check: bool = True) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=check)


def git_mv(src: str, dst: str) -> None:
    src_p = ROOT / src
    dst_p = ROOT / dst
    if not src_p.exists():
        print(f"skip missing: {src}")
        return
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    ret = subprocess.run(
        ["git", "mv", src, dst], cwd=ROOT, capture_output=True, text=True
    )
    if ret.returncode == 0:
        print(f"+ git mv {src} {dst}")
        return
    if "cross-device" in (ret.stderr or "").lower() or ret.returncode == 128:
        shutil.move(str(src_p), str(dst_p))
        run(["git", "add", "-A", dst, src], check=False)
        print(f"+ mv+git add {src} -> {dst}")
        return
    ret.check_returncode()


def move_tree() -> None:
    (ROOT / "nntile/src").mkdir(parents=True, exist_ok=True)
    (ROOT / "nntile/include/nntile").mkdir(parents=True, exist_ok=True)
    (ROOT / "nntile/tests").mkdir(parents=True, exist_ok=True)

    for sub in ("kernel", "starpu", "tile", "tensor", "logger"):
        git_mv(f"src/core/{sub}", f"nntile/src/{sub}")
        git_mv(f"include/nntile/core/{sub}", f"nntile/include/nntile/{sub}")

    for name in (
        "base_types.hh",
        "constants.hh",
        "context.hh",
        "defs.h.in",
        "kernel.hh",
        "logger.hh",
        "starpu_c.hh",
        "starpu.hh",
        "tensor.hh",
        "tile.hh",
    ):
        git_mv(f"include/nntile/core/{name}", f"nntile/include/nntile/{name}")

    git_mv("src/core/context.cc", "nntile/src/context.cc")

    graph_renames = {
        "tile": "tile_graph",
        "tensor": "tensor_graph",
        "nn": "nn_graph",
    }
    for old, new in graph_renames.items():
        git_mv(f"src/graph/{old}", f"nntile/src/{new}")
        git_mv(f"include/nntile/graph/{old}", f"nntile/include/nntile/{new}")

    for sub in ("module", "model", "optim", "io", "dataset"):
        git_mv(f"src/graph/{sub}", f"nntile/src/{sub}")
        git_mv(f"include/nntile/graph/{sub}", f"nntile/include/nntile/{sub}")

    for name in (
        "common.hh",
        "dtype.hh",
        "io.hh",
        "kv_cache.hh",
        "module.hh",
        "nn.hh",
        "optim.hh",
        "runtime.hh",
        "tensor.hh",
        "tile.hh",
    ):
        src = f"include/nntile/graph/{name}"
        if name in ("tensor.hh", "tile.hh"):
            dst = f"nntile/include/nntile/{name.replace('.hh', '_graph.hh')}"
        else:
            dst = f"nntile/include/nntile/{name}"
        if (ROOT / src).exists():
            git_mv(src, dst)

    for name in ("dtype.cc", "kv_cache.cc", "runtime.cc"):
        git_mv(f"src/graph/{name}", f"nntile/src/{name}")

    git_mv("include/nntile.hh", "nntile/include/nntile.hh")
    git_mv("include/nntile/core.hh", "nntile/include/nntile/core.hh")
    git_mv("include/nntile/graph.hh", "nntile/include/nntile/graph.hh")

    git_mv("tests/core", "nntile/tests/eager")

    test_renames = {
        "tile": "tile_graph",
        "tensor": "tensor_graph",
        "nn": "nn_graph",
    }
    for old, new in test_renames.items():
        git_mv(f"tests/graph/{old}", f"nntile/tests/{new}")

    for sub in ("module", "model", "io"):
        git_mv(f"tests/graph/{sub}", f"nntile/tests/{sub}")

    for name in ("context_fixture.hh", "nn.cc"):
        git_mv(f"tests/graph/{name}", f"nntile/tests/{name}")

    git_mv("examples", "nntile/examples")

    # Remove empty dirs if git left them
    for p in (
        "src/core",
        "src/graph",
        "src",
        "include/nntile/core",
        "include/nntile/graph",
        "include/nntile",
        "include",
        "tests/core",
        "tests/graph",
        "tests",
    ):
        d = ROOT / p
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()


def rewrite_text(text: str) -> str:
    # Includes: most specific first
    inc_pairs = [
        (r"#include\s*([<\"])nntile/graph/tile/", r"#include \1nntile/tile_graph/"),
        (r"#include\s*([<\"])nntile/graph/tensor/", r"#include \1nntile/tensor_graph/"),
        (r"#include\s*([<\"])nntile/graph/nn/", r"#include \1nntile/nn_graph/"),
        (r"#include\s*([<\"])nntile/core/", r"#include \1nntile/"),
        (r"#include\s*([<\"])nntile/graph/", r"#include \1nntile/"),
        (r"#include\s*([<\"])nntile/core\.hh", r"#include \1nntile/core.hh"),
        (r"#include\s*([<\"])nntile/graph\.hh", r"#include \1nntile/graph.hh"),
    ]
    for pat, repl in inc_pairs:
        text = re.sub(pat, repl, text)

    # Umbrella / install paths in cmake lists
    text = text.replace("nntile/core/", "nntile/")
    text = text.replace("nntile/graph/tile/", "nntile/tile_graph/")
    text = text.replace("nntile/graph/tensor/", "nntile/tensor_graph/")
    text = text.replace("nntile/graph/nn/", "nntile/nn_graph/")
    text = text.replace("nntile/graph/", "nntile/")

    # Namespaces: longest prefixes first
    ns_pairs = [
        ("nntile::graph::model::", "nntile::model::"),
        ("nntile::graph::module::", "nntile::module::"),
        ("nntile::graph::optim::", "nntile::optim::"),
        ("nntile::graph::io::", "nntile::io::"),
        ("nntile::graph::dataset::", "nntile::dataset::"),
        ("nntile::graph::tile_graph::", "nntile::tile_graph::"),
        ("nntile::graph::tensor::", "nntile::tensor_graph::"),
        ("nntile::graph::tile_lower::", "nntile::tensor_graph::tile_lower::"),
        ("nntile::core::kernel::", "nntile::kernel::"),
        ("nntile::core::starpu::", "nntile::starpu::"),
        ("nntile::core::tile::", "nntile::tile::"),
        ("nntile::core::tensor::", "nntile::tensor::"),
        ("nntile::core::logger::", "nntile::logger::"),
        ("namespace nntile::graph::model", "namespace nntile::model"),
        ("namespace nntile::graph::module", "namespace nntile::module"),
        ("namespace nntile::graph::optim", "namespace nntile::optim"),
        ("namespace nntile::graph::tile_graph", "namespace nntile::tile_graph"),
        ("namespace nntile::graph::tensor", "namespace nntile::tensor_graph"),
        ("namespace nntile::core::kernel", "namespace nntile::kernel"),
        ("namespace nntile::core::starpu", "namespace nntile::starpu"),
        ("namespace nntile::core::tile", "namespace nntile::tile"),
        ("namespace nntile::core::tensor", "namespace nntile::tensor"),
        ("namespace nntile::core::logger", "namespace nntile::logger"),
        ("namespace nntile::graph", "namespace nntile"),
        ("namespace nntile::core", "namespace nntile"),
        ("} // namespace nntile::graph::model", "} // namespace nntile::model"),
        ("} // namespace nntile::graph::module", "} // namespace nntile::module"),
        ("} // namespace nntile::graph::tile_graph", "} // namespace nntile::tile_graph"),
        ("} // namespace nntile::graph::tensor", "} // namespace nntile::tensor_graph"),
        ("} // namespace nntile::graph", "} // namespace nntile"),
        ("} // namespace nntile::core", "} // namespace nntile"),
        ("nntile::graph::", "nntile::"),
        ("nntile::core::", "nntile::"),
    ]
    for old, new in ns_pairs:
        text = text.replace(old, new)

    # Source tree paths in comments / cmake
    text = text.replace("src/core/", "nntile/src/")
    text = text.replace("src/graph/tile/", "nntile/src/tile_graph/")
    text = text.replace("src/graph/tensor/", "nntile/src/tensor_graph/")
    text = text.replace("src/graph/nn/", "nntile/src/nn_graph/")
    text = text.replace("src/graph/", "nntile/src/")
    text = text.replace("tests/core/", "nntile/tests/eager/")
    text = text.replace("tests/graph/tile/", "nntile/tests/tile_graph/")
    text = text.replace("tests/graph/tensor/", "nntile/tests/tensor_graph/")
    text = text.replace("tests/graph/nn/", "nntile/tests/nn_graph/")
    text = text.replace("tests/graph/", "nntile/tests/")

    text = text.replace("nntile_core", "nntile")
    text = text.replace("nntile_graph", "nntile")
    # Restore accidental double replacements in historical refs
    text = text.replace("BUILD_nntile", "BUILD_NNTILE")

    return text


def rewrite_files() -> None:
    skip_dirs = {
        ".git",
        "build",
        "external",
        "uploads",
    }
    for path in ROOT.rglob("*"):
        if path.is_dir():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name not in (
            "CMakeLists.txt",
            "pyproject.toml",
        ):
            continue
        if "migrate_plan_b.py" in str(path):
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        updated = rewrite_text(original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            print("rewrote", path.relative_to(ROOT))


def main() -> int:
    if "--move-only" in sys.argv:
        move_tree()
        return 0
    if "--rewrite-only" in sys.argv:
        rewrite_files()
        return 0
    move_tree()
    rewrite_files()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
