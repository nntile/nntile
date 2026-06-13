#!/usr/bin/env python3
"""Migrate NNTile tests from Fortran-labeled to C-order shape conventions."""

from __future__ import annotations

import re
import sys
from pathlib import Path

TEST_ROOTS = [
    Path("nntile/tests/core"),
    Path("nntile/tests/kernel"),
    Path("nntile/tests/starpu"),
    Path("nntile/tests/tensor"),
    Path("nntile/tests/tile"),
    Path("nntile/tests/nn"),
    Path("nntile/tests/module"),
    Path("nntile/tests/model"),
]

FIBER_FIXES = [
    (
        """    out.push_back(tensor_shape[axis]);
    for (Index i = 0; i < batch_ndim; ++i)
    {
        out.push_back(tensor_shape[tensor_shape.size() - batch_ndim + i]);
    }""",
        """    for (Index i = 0; i < batch_ndim; ++i)
    {
        out.push_back(tensor_shape[i]);
    }
    out.push_back(tensor_shape[axis]);""",
    ),
    (
        """    out_shape.push_back(x_shape[axis]);
    for (Index i = 0; i < batch_ndim; ++i)
    {
        out_shape.push_back(x_shape[x_shape.size() - batch_ndim + i]);
    }""",
        """    for (Index i = 0; i < batch_ndim; ++i)
    {
        out_shape.push_back(x_shape[i]);
    }
    out_shape.push_back(x_shape[axis]);""",
    ),
    (
        """    out.push_back(dst_shape[axis]);
    for (Index i = 0; i < batch_ndim; ++i)
    {
        out.push_back(dst_shape[dst_shape.size() - batch_ndim + i]);
    }""",
        """    for (Index i = 0; i < batch_ndim; ++i)
    {
        out.push_back(dst_shape[i]);
    }
    out.push_back(dst_shape[axis]);""",
    ),
]

# Introducers whose following {a, b, ...} is a tensor shape literal.
SHAPE_INTRO_RE = re.compile(
    r"(?:"
    r"Tile<[^>]+>\(|"
    r"(?:graph|g|graph2|other|nn_graph|nng)\.(?:data|tensor)\(|"
    r"(?:std::)?vector<Index>\s*\{|"
    r"std::vector<Index>\s+\w+\s*=\s*\{|"
    r"torch::full\(|"
    r"\.reshape\(|"
    r"torch::from_blob\([^;]+,\s*\{|"
    r"==\s*(?:\(std::vector<Index>\))?\{|"
    r"REQUIRE\([^;]*\{"
    r")"
)


def reverse_brace_list(inner: str) -> str:
    parts = [p.strip() for p in inner.split(",")]
    if len(parts) < 2:
        return inner
    return ", ".join(reversed(parts))


def reverse_shape_literals_once(text: str) -> str:
    """Single pass: reverse {..} lists that follow shape introducers."""

    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        m = SHAPE_INTRO_RE.match(text, i)
        if not m:
            out.append(text[i])
            i += 1
            continue
        out.append(m.group(0))
        i = m.end()
        while i < n and text[i] in " \t\n(":
            out.append(text[i])
            i += 1
        if i >= n or text[i] != "{":
            continue
        j = i + 1
        depth = 1
        while j < n and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth != 0:
            continue
        inner = text[i + 1 : j - 1]
        if "," in inner:
            rev = reverse_brace_list(inner)
            out.append("{" + rev + "}")
        else:
            out.append(text[i:j])
        i = j
    return "".join(out)


def swap_axis_constants(text: str) -> str:
    if "axis_0" not in text and "axis_1" not in text:
        return text
    text = text.replace("axis_0", "axis__TMP__")
    text = text.replace("axis_1", "axis_0")
    return text.replace("axis__TMP__", "axis_1")


def remap_core_axis_checks(text: str) -> str:
    return re.sub(
        r"check<T>\(([^,]+),\s*([^,]+),\s*(\d+)\)",
        lambda m: f"check<T>({m.group(1)}, {m.group(2)}, {2 - int(m.group(3))})"
        if int(m.group(3)) in (0, 1, 2)
        else m.group(0),
        text,
    )


def remove_colmajor_conversions(text: str) -> str:
    text = re.sub(
        r"colmajor_to_rowmajor\(([^,]+),\s*\{[^}]+\}\)",
        r"\1",
        text,
    )
    text = re.sub(
        r"std::vector<float>\s+\w+_rowmajor\s*=\s*(\w+);\s*\n",
        "",
        text,
    )
    text = re.sub(
        r"std::vector<float>\s+\w+_data_rowmajor\s*=\s*(\w+);\s*\n",
        "",
        text,
    )
    text = re.sub(r"\w+_rowmajor\.data\(\)", lambda m: m.group(0).replace("_rowmajor", ""), text)
    text = re.sub(
        r"\w+_data_rowmajor\.data\(\)",
        lambda m: m.group(0).replace("_data_rowmajor", "_data"),
        text,
    )
    text = text.replace("nntile_out_colmajor", "nntile_out")
    text = re.sub(r"nntile_grad_(\w+)_colmajor", r"nntile_grad_\1", text)
    text = re.sub(
        r"std::vector<float>\s+nntile_out_rowmajor\s*=\s*nntile_out;\s*\n"
        r"\s*std::vector<float>\s+nntile_out\s*=\s*"
        r"permute_rowmajor\([^;]+;\s*\n",
        "",
        text,
    )
    text = re.sub(
        r"std::vector<float>\s+nntile_grad_(\w+)_rowmajor\s*=\s*"
        r"nntile_grad_\1;\s*\n\s*std::vector<float>\s+nntile_grad_\1\s*=\s*"
        r"permute_rowmajor\([^;]+;\s*\n",
        "",
        text,
    )
    text = re.sub(
        r"auto a_batched = a_pt\.permute\(\{2, 0, 1\}\);\s*\n"
        r"\s*auto b_batched = b_pt\.permute\(\{2, 0, 1\}\);\s*\n"
        r"\s*auto out_pt = (gemm_alpha \* )?torch::bmm\(a_batched, b_batched\)",
        r"auto out_pt = \1torch::bmm(a_pt, b_pt)",
        text,
    )
    text = re.sub(
        r"nntile_grad_a, a_pt\.grad\(\)\.permute\(\{2, 0, 1\}\)",
        "nntile_grad_a, a_pt.grad()",
        text,
    )
    text = re.sub(
        r"nntile_grad_b, b_pt\.grad\(\)\.permute\(\{2, 0, 1\}\)",
        "nntile_grad_b, b_pt.grad()",
        text,
    )
    if "colmajor_to_rowmajor" not in text:
        text = re.sub(r"using nntile::test::colmajor_to_rowmajor;\s*\n", "", text)
    if "permute_rowmajor" not in text:
        text = re.sub(r"using nntile::test::permute_rowmajor;\s*\n", "", text)
    return text


def fix_concat_reference(text: str) -> str:
    text = text.replace("fortran_dense_linear_index", "dense_linear_index")
    text = text.replace("fortran_tile_linear_to_index", "tile_linear_to_index")
    text = text.replace("reference_concat_fortran", "reference_concat_dense")
    text = text.replace(
        "Fortran flat layout (same as bind_data / get_output)",
        "C-order dense layout (same as bind_data / get_output)",
    )
    text = text.replace(
        "TensorGraph concat matches Fortran reference",
        "TensorGraph concat matches dense reference",
    )
    return text


def fix_mask_scalar_names(text: str) -> str:
    if "mask_scalar" not in str(text) and "nrows" not in text:
        return text
    text = re.sub(r"\bnrows\b", "nslow", text)
    return re.sub(r"\bncols\b", "nfast", text)


def fix_model_helpers(text: str) -> str:
    subs = [
        ("g.tensor({n_seq, n_batch}", "g.tensor({n_batch, n_seq}"),
        ("g.tensor({half, n_seq, n_batch}", "g.tensor({n_batch, n_seq, half}"),
        ("g.tensor({hidden, n_seq, n_batch}", "g.tensor({n_batch, n_seq, hidden}"),
        ("info.shape[0] != n_seq ||\n        info.shape[1] != n_batch",
         "info.shape[0] != n_batch ||\n        info.shape[1] != n_seq"),
    ]
    for old, new in subs:
        text = text.replace(old, new)
    return text


def fix_comments(text: str) -> str:
    text = text.replace(
        "column-major for\n    // NNTile", "C-order for NNTile"
    )
    text = text.replace("column-major for NNTile", "C-order for NNTile")
    return text


def process_file(path: Path, dry: bool) -> bool:
    original = path.read_text()
    text = original
    path_s = str(path).replace("\\", "/")

    text = fix_concat_reference(text)
    text = fix_mask_scalar_names(text)
    for old, new in FIBER_FIXES:
        text = text.replace(old, new)
    text = reverse_shape_literals_once(text)

    if "/tests/model/" in path_s:
        text = fix_model_helpers(text)
    if "/tests/core/" in path_s or "/tests/kernel/" in path_s:
        text = remap_core_axis_checks(text)
    elif any(f"/tests/{d}/" in path_s for d in ("tensor", "tile", "nn")):
        text = swap_axis_constants(text)

    text = remove_colmajor_conversions(text)
    text = fix_comments(text)

    if text != original:
        if not dry:
            path.write_text(text)
        return True
    return False


def main(argv: list[str]) -> int:
    dry = "--dry-run" in argv
    files: list[Path] = []
    for root in TEST_ROOTS:
        if root.exists():
            files.extend(sorted(root.rglob("*.cc")))
            files.extend(sorted(root.rglob("*.hh")))
    changed = [p for p in files if process_file(p, dry)]
    for p in changed:
        print("migrated" + (" (dry)" if dry else ""), p)
    print(f"total changed: {len(changed)} / {len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
