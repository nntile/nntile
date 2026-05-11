#!/usr/bin/env python3
"""Strip output_name from tensor-level graph ops (headers + sources)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TENSOR_OPS_HH = ROOT / "include/nntile/graph/tensor/ops"
TENSOR_OPS_CC = ROOT / "src/graph/tensor/ops"

_GEMM_PATCH_RE = (
    r"(TensorGraph::TensorNode\* gemm\(\s*\n\s*TensorGraph::TensorNode\* a,"
    r"\s*\n\s*TensorGraph::TensorNode\* b,)\s*\n\s*const std::string& "
    r"output_name,\s*\n\s*(Scalar alpha,)"
)
_TRANSPOSE_PATCH_RE = (
    r"(TensorGraph::TensorNode\* transpose\(\s*\n\s*Scalar alpha,\s*\n\s*"
    r"TensorGraph::TensorNode\* src,)\s*\n\s*const std::string& output_name,"
    r"\s*\n\s*(Index ndim)"
)
_EMBEDDING_PATCH_RE = (
    r"(TensorGraph::TensorNode\* embedding\(TensorGraph::TensorNode\* index,"
    r"\s*\n\s*TensorGraph::TensorNode\* vocab,)\s*\n\s*const std::string& "
    r"output_name,\s*\n\s*(Index axis)"
)


def patch_gemm(text: str) -> str:
    text = re.sub(
        _GEMM_PATCH_RE,
        r"\1\n    \2",
        text,
        count=1,
    )
    return text


def patch_transpose(text: str) -> str:
    text = re.sub(
        _TRANSPOSE_PATCH_RE,
        r"\1\n    \2",
        text,
        count=1,
    )
    return text


def patch_embedding(text: str) -> str:
    text = re.sub(
        _EMBEDDING_PATCH_RE,
        r"\1\n    \2",
        text,
        count=1,
    )
    return text


def strip_generic(text: str) -> str:
    text = text.replace(", const std::string& output_name)", ")")
    text = text.replace(", const std::string& output_name,", ",")
    text = re.sub(r"//! @param output_name[^\n]*\n", "", text)
    return text


def process_file(path: Path) -> None:
    text = path.read_text()
    orig = text
    base = path.name
    if base == "gemm.hh" or base == "gemm.cc":
        text = patch_gemm(text)
    if base == "transpose.hh" or base == "transpose.cc":
        text = patch_transpose(text)
    if base == "embedding.hh" or base == "embedding.cc":
        text = patch_embedding(text)
    text = strip_generic(text)
    if path.suffix == ".cc":
        text = re.sub(r"\boutput_name\b", '""', text)
    if text != orig:
        path.write_text(text)
        print("updated", path.relative_to(ROOT))


def main() -> None:
    for d in (TENSOR_OPS_HH, TENSOR_OPS_CC):
        for path in sorted(d.glob("*")):
            if path.suffix in {".hh", ".cc"}:
                process_file(path)


if __name__ == "__main__":
    main()
