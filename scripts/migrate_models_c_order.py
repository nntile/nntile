#!/usr/bin/env python3
"""Apply C-order virtual shape labels to NNGraph model sources.

Keeps graph_api forward op patterns (trailing-batch tensor GEMM) and:
- Reverses ``graph.tensor({...})`` / ``graph_->tensor({...})`` literals
- Maps legacy Fortran ``add_fiber`` / ``layer_norm`` axes to C-order axes
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def reverse_shape_literal(inner: str) -> str:
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    return ", ".join(reversed(parts))


def migrate_shapes(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        prefix = m.group(1)
        inner = m.group(2)
        return f"{prefix}{{{reverse_shape_literal(inner)}}}"

    text = re.sub(
        r"((?:graph|graph_)->tensor\(\{)([^}]+)(\})",
        repl,
        text,
    )
    text = re.sub(
        r"(g\.tensor\(\{)([^}]+)(\})",
        repl,
        text,
    )
    return text


def migrate_layer_norm_axes(text: str) -> str:
    # Normalize over hidden (last C axis) for 3D activations [batch, seq, hidden].
    return re.sub(
        r"(LayerNorm|layer_norm_)\([^)]*,\s*"
        r"(?:config\.(?:hidden_size|d_model)|normalized_shape[^,]*),\s*)-1,",
        r"\1(\2 2,",
        text,
    )


def migrate_add_fiber_axes(text: str) -> str:
    # Legacy Fortran axis 0 on ndim-D tensor -> C axis (D - 1).
    replacements = {
        ", 0, 1)": ", 3, 1)",  # 4D Q/K/V bias, batch_ndim=1
        ", 0, 0)": ", 2, 0)",  # 3D feature bias, batch_ndim=0
    }
    for old, new in replacements.items():
        text = text.replace(f"add_fiber(1.0, o_bias_, 1.0, out{old}", f"add_fiber(1.0, o_bias_, 1.0, out{new}")
        text = text.replace(f"add_fiber(1.0, fc1_bias_, 1.0, hidden{old}", f"add_fiber(1.0, fc1_bias_, 1.0, hidden{new}")
        text = text.replace(f"add_fiber(1.0, fc2_bias_, 1.0, out{old}", f"add_fiber(1.0, fc2_bias_, 1.0, out{new}")
    text = text.replace(
        "add_fiber(1.0, q_bias_, 1.0, q, 0, 1)",
        "add_fiber(1.0, q_bias_, 1.0, q, 3, 1)",
    )
    text = text.replace(
        "add_fiber(1.0, k_bias_, 1.0, k, 0, 1)",
        "add_fiber(1.0, k_bias_, 1.0, k, 3, 1)",
    )
    text = text.replace(
        "add_fiber(1.0, v_bias_, 1.0, v, 0, 1)",
        "add_fiber(1.0, v_bias_, 1.0, v, 3, 1)",
    )
    return text


def migrate_file(path: Path) -> bool:
    original = path.read_text()
    updated = migrate_shapes(original)
    updated = migrate_layer_norm_axes(updated)
    updated = migrate_add_fiber_axes(updated)
    if updated != original:
        path.write_text(updated)
        return True
    return False


def main() -> int:
    roots = [
        Path("nntile/src/model"),
        Path("nntile/include/nntile/model"),
    ]
    changed = 0
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix not in {".cc", ".hh"}:
                continue
            if migrate_file(path):
                print(f"updated {path}")
                changed += 1
    print(f"done, {changed} files changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
