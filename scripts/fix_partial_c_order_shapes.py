#!/usr/bin/env python3
"""Fix partially migrated C-order shape literals in tests.

The first migration pass reversed only the first Tile<T>({...}) on a line,
leaving sibling tensors at Fortran order. This script normalizes common
Fortran->C shape reversals across test files.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOTS = [
    Path("nntile/tests/core"),
    Path("nntile/tests/kernel"),
    Path("nntile/tests/starpu"),
    Path("nntile/tests/tensor"),
    Path("nntile/tests/tile"),
    Path("nntile/tests/nn"),
    Path("nntile/tests/module"),
]

# Fortran shape -> C-order shape (same flat layout).
SHAPE_MAP = {
    "{2, 3, 4}": "{4, 3, 2}",
    "{3, 4, 5}": "{5, 4, 3}",
    "{3, 4, 6}": "{6, 4, 3}",
    "{3, 5, 6}": "{6, 5, 3}",
    "{2, 3, 4, 5}": "{5, 4, 3, 2}",
    "{2, 2, 3}": "{3, 2, 2}",
    "{2, 3}": "{3, 2}",
    "{3, 5}": "{5, 3}",
    "{4, 3}": "{3, 4}",
    "{5, 3}": "{3, 5}",
    "{5, 4}": "{4, 5}",
    "{4, 5}": "{5, 4}",
    "{5, 3, 4}": "{4, 3, 5}",
    "{4, 3, 2}": "{4, 3, 2}",  # noop anchor
}


def fix_transpose(text: str) -> str:
    # After C-order: src was {3,5} -> {5,3}, dst was {5,3} -> {3,5}
    text = re.sub(
        r"Tile<T> src\(\{5, 3\}\), dst\(\{5, 3\}\), dst_ref\(\{5, 3\}\)",
        "Tile<T> src({5, 3}), dst({3, 5}), dst_ref({3, 5})",
        text,
    )
    return text


def fix_file(path: Path) -> bool:
    original = path.read_text()
    text = original
    for old, new in SHAPE_MAP.items():
        if old == new:
            continue
        text = text.replace(old, new)
    text = fix_transpose(text)
    if text != original:
        path.write_text(text)
        return True
    return False


def main() -> None:
    changed = 0
    for root in ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.cc")):
            if fix_file(path):
                print("fixed", path)
                changed += 1
        for path in sorted(root.rglob("*.hh")):
            if fix_file(path):
                print("fixed", path)
                changed += 1
    print(f"total changed: {changed}")


if __name__ == "__main__":
    main()
