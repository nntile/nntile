#!/usr/bin/env python3
"""Fix tile slice/rope test shapes for C-order migration.

Slice tensors drop one axis from the full tensor; they are not a full shape
reversal of the Fortran-era slice shape.
"""

from __future__ import annotations

from pathlib import Path

FIXES: dict[str, list[tuple[str, str]]] = {
    "add_slice.cc": [
        (
            "const std::vector<Index> t1s = {5, 4}, t2s = {5, 4, 3}, ds = {5, 4, 3};\n"
            "    const Index n1 = 20, n2 = 60;",
            "const std::vector<Index> t1s = {4, 3}, t2s = {5, 4, 3}, ds = {5, 4, 3};\n"
            "    const Index n1 = 12, n2 = 60;",
        ),
    ],
    "add_slice_inplace.cc": [
        (
            "const std::vector<Index> t1s = {3, 5}, t2s = {5, 4, 3};",
            "const std::vector<Index> t1s = {4, 3}, t2s = {5, 4, 3};",
        ),
    ],
    "scale_slice.cc": [
        (
            "const std::vector<Index> t1s = {3, 5}, t2s = {5, 4, 3};",
            "const std::vector<Index> t1s = {4, 3}, t2s = {5, 4, 3};",
        ),
    ],
    "norm_slice.cc": [
        (
            "const std::vector<Index> t1s = {5, 4}, t2s = {5, 4, 3}, ds = {5, 4, 3};",
            "const std::vector<Index> t1s = {4, 3}, t2s = {5, 4, 3}, ds = {5, 4, 3};",
        ),
    ],
    "norm_slice_inplace.cc": [
        (
            "const std::vector<Index> t1s = {3, 5}, t2s = {5, 4, 3};",
            "const std::vector<Index> t1s = {4, 3}, t2s = {5, 4, 3};",
        ),
    ],
    "sum_slice.cc": [
        (
            "const std::vector<Index> ss = {5, 4}, ds = {5, 4, 3};",
            "const std::vector<Index> ss = {4, 3}, ds = {5, 4, 3};",
        ),
    ],
    "sumprod_slice.cc": [
        (
            "const std::vector<Index> s1s = {5, 4}, s2s = {5, 4, 3}, ds = {5, 4, 3};",
            "const std::vector<Index> s1s = {4, 3}, s2s = {5, 4, 3}, ds = {5, 4, 3};",
        ),
    ],
    "multiply_slice.cc": [
        (
            "const std::vector<Index> ss = {5, 4}, ds = {5, 4, 3};",
            "const std::vector<Index> ss = {4, 3}, ds = {5, 4, 3};",
        ),
    ],
    "rope.cc": [
        (
            "const std::vector<Index> sh = {2}, tsh = {4,5};",
            "const std::vector<Index> sh = {2}, tsh = {5, 4};",
        ),
    ],
    "rope_backward.cc": [
        (
            "const std::vector<Index> sh = {2}, tsh = {4,5};",
            "const std::vector<Index> sh = {2}, tsh = {5, 4};",
        ),
    ],
}


def main() -> None:
    root = Path("nntile/tests/tile/ops")
    changed = 0
    for name, reps in FIXES.items():
        path = root / name
        if not path.exists():
            continue
        text = path.read_text()
        original = text
        for old, new in reps:
            text = text.replace(old, new)
        if text != original:
            path.write_text(text)
            print("fixed", path)
            changed += 1
    print(f"total changed: {changed}")


if __name__ == "__main__":
    main()
