#!/usr/bin/env python3
"""Reverse Tile({...}) shape literals missed by first migration pass."""

from __future__ import annotations

import re
from pathlib import Path

ROOTS = [
    Path("nntile/tests/core"),
    Path("nntile/tests/kernel"),
    Path("nntile/tests/starpu"),
]


def reverse_inner(inner: str) -> str:
    parts = [p.strip() for p in inner.split(",")]
    if len(parts) < 2:
        return inner
    return ", ".join(reversed(parts))


def fix_file(path: Path) -> bool:
    text = path.read_text()
    orig = text

    def repl(m: re.Match[str]) -> str:
        inner = m.group(1)
        if "," not in inner:
            return m.group(0)
        return "Tile<T>({" + reverse_inner(inner) + "}"

    text = re.sub(r"Tile<T>\(\{([^{}]+)\}\)", repl, text)
    text = re.sub(
        r"Tile<T> (\w+)\(\{([^{}]+)\}\)",
        lambda m: f"Tile<T> {m.group(1)}({{{reverse_inner(m.group(2))}}})"
        if "," in m.group(2)
        else m.group(0),
        text,
    )
    text = re.sub(
        r"Tile<T> dst\(\{([^{}]+)\}\)",
        lambda m: f"Tile<T> dst({{{reverse_inner(m.group(1))}}})"
        if "," in m.group(1)
        else m.group(0),
        text,
    )
    text = re.sub(
        r"Tile<T> (src\w*)\(\{([^{}]+)\}\)",
        lambda m: f"Tile<T> {m.group(1)}({{{reverse_inner(m.group(2))}}})"
        if "," in m.group(1)
        else m.group(0),
        text,
    )
    text = re.sub(
        r"Tile<T> (mat\w+)\(\{([^{}]+)\}\)",
        lambda m: f"Tile<T> {m.group(1)}({{{reverse_inner(m.group(2))}}})"
        if "," in m.group(2)
        else m.group(0),
        text,
    )
    text = re.sub(
        r"Tile<T> (A|B|C|D)\(\{([^{}]+)\}\)",
        lambda m: f"Tile<T> {m.group(1)}({{{reverse_inner(m.group(2))}}})"
        if "," in m.group(2)
        else m.group(0),
        text,
    )
    text = re.sub(
        r"Tile<T> (mask|data)\(\{([^{}]+)\}\)",
        lambda m: f"Tile<T> {m.group(1)}({{{reverse_inner(m.group(2))}}})"
        if "," in m.group(2)
        else m.group(0),
        text,
    )

    if text != orig:
        path.write_text(text)
        return True
    return False


def main() -> None:
    n = 0
    for root in ROOTS:
        for path in sorted(root.rglob("*.cc")):
            if fix_file(path):
                print("fixed", path)
                n += 1
    print("total", n)


if __name__ == "__main__":
    main()
