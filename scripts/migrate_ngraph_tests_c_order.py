#!/usr/bin/env python3
"""Migrate NNGraph test shape literals from Fortran to C-order labels."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def reverse_shape_literal(inner: str) -> str:
    """Reverse comma-separated Index literals inside braces."""
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    return ", ".join(reversed(parts))


def migrate_shapes_in_text(text: str) -> str:
    # g.tensor({...}) and graph_->tensor({...})
    def repl_tensor(m: re.Match[str]) -> str:
        prefix = m.group(1)
        inner = m.group(2)
        return f"{prefix}{{{reverse_shape_literal(inner)}}}"

    text = re.sub(
        r"((?:g|graph_)->tensor\(\{)([^}]+)(\})",
        repl_tensor,
        text,
    )
    # std::vector<Index>{...} shape literals (conservative)
    def repl_vec(m: re.Match[str]) -> str:
        inner = m.group(1)
        if "shape" not in m.string[max(0, m.start() - 40) : m.start()].lower():
            return m.group(0)
        return f"std::vector<Index>{{{reverse_shape_literal(inner)}}}"

    text = re.sub(
        r"std::vector<Index>\{([^}]+)\}",
        repl_vec,
        text,
    )
    return text


def migrate_file(path: Path) -> bool:
    original = path.read_text()
    updated = migrate_shapes_in_text(original)
    if updated != original:
        path.write_text(updated)
        return True
    return False


def main() -> int:
    roots = [
        Path("nntile/tests/nn/ops"),
        Path("nntile/tests/nn"),
        Path("nntile/tests/model"),
        Path("nntile/tests/module"),
    ]
    changed = 0
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.cc")):
            if migrate_file(path):
                print(f"updated {path}")
                changed += 1
        for path in sorted(root.rglob("*.hh")):
            if migrate_file(path):
                print(f"updated {path}")
                changed += 1
    print(f"done, {changed} files changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
