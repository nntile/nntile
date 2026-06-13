#!/usr/bin/env python3
"""Second-pass C-order fixes for tests after partial migration."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path("nntile/tests")

# Fortran [in, batch] or [feature, batch] -> C [batch, feature]
INPUT_2D = [
    ("g.tensor({3, 2}", "g.tensor({2, 3}"),
    ("g.tensor({4, 3, 2}", "g.tensor({2, 3, 4}"),
]

# Linear weight [in, out] -> [out, in]
WEIGHT_LINEAR = [
    ("g.tensor({3, 4}", "g.tensor({4, 3}"),
    ("std::vector<Index>({3, 4})", "std::vector<Index>({4, 3})"),
]

# Model activations [hidden, seq, batch] -> [batch, seq, hidden]
MODEL_SHAPES = [
    ("{fx.hidden, fx.seq, fx.batch}", "{fx.batch, fx.seq, fx.hidden}"),
    ("{fx.hidden, fx.dec_seq, fx.batch}", "{fx.batch, fx.dec_seq, fx.hidden}"),
    ("{fx.hidden, fx.enc_seq, fx.batch}", "{fx.batch, fx.enc_seq, fx.hidden}"),
]

TRANSPOSE_FIXES = [
    (
        "Tile<T> src({3, 5}), dst({3, 5}), dst_ref({3, 5})",
        "Tile<T> src({5, 3}), dst({3, 5}), dst_ref({3, 5})",
    ),
    (
        "const std::vector<Index> sshape = {3, 5};\n"
        "    const std::vector<Index> dshape = {3, 5};",
        "const std::vector<Index> sshape = {5, 3};\n"
        "    const std::vector<Index> dshape = {3, 5};",
    ),
]

FIBER_FIX = (
    "const std::vector<Index> full = {5, 4, 3};\n"
    "    const std::vector<Index> fib = {5};",
    "const std::vector<Index> full = {5, 4, 3};\n"
    "    const std::vector<Index> fib = {3};",
)

AXIS_DESCRIPTOR_FIXES = [
    ("REQUIRE(x->axis(0)->extent == 4);", "REQUIRE(x->axis(0)->extent == 5);"),
    ("REQUIRE(x->axis(1)->extent == 5);", "REQUIRE(x->axis(1)->extent == 4);"),
]


def apply_replacements(path: Path, replacements: list[tuple[str, str]]) -> bool:
    text = path.read_text()
    original = text
    for old, new in replacements:
        text = text.replace(old, new)
    if text != original:
        path.write_text(text)
        return True
    return False


def fix_file(path: Path) -> bool:
    changed = False
    reps: list[tuple[str, str]] = []
    if "model/" in path.as_posix():
        reps.extend(MODEL_SHAPES)
    if "module/" in path.as_posix():
        reps.extend(INPUT_2D)
        reps.extend(WEIGHT_LINEAR)
        # grad / input shape assertions in linear.cc
        reps.append(
            ("std::vector<Index>({3, 2})", "std::vector<Index>({2, 3})")
        )
        reps.append(
            ("g.tensor({2, 5}", "g.tensor({5, 2}")
        )
    if path.name == "transpose.cc":
        reps.extend(TRANSPOSE_FIXES)
    if path.name == "axis_descriptor.cc":
        reps.extend(AXIS_DESCRIPTOR_FIXES)
    if path.name in {
        "add_fiber.cc",
        "add_fiber_inplace.cc",
        "multiply_fiber.cc",
        "multiply_fiber_inplace.cc",
        "scale_fiber.cc",
    } and "tile/ops" in path.as_posix():
        reps.append(FIBER_FIX)
    if reps:
        changed = apply_replacements(path, reps) or changed
    return changed


def main() -> None:
    changed = 0
    for path in sorted(ROOT.rglob("*.cc")):
        if fix_file(path):
            print("fixed", path)
            changed += 1
    print(f"total changed: {changed}")


if __name__ == "__main__":
    main()
