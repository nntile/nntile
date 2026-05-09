#!/usr/bin/env python3
"""Strip output_name from NN graph op headers and sources."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NN_HH = ROOT / "include/nntile/graph/nn/ops"
NN_CC = ROOT / "src/graph/nn/ops"

_GEMM_HDR_RE = (
    r"(NNGraph::TensorNode\* gemm\(\s*\n\s*NNGraph::TensorNode\* a,"
    r"\s*\n\s*NNGraph::TensorNode\* b,)\s*\n\s*const std::string& output_name,"
    r"\s*\n\s*(Scalar alpha,)"
)
_TRANSPOSE_HDR_RE = (
    r"(NNGraph::TensorNode\* transpose\(\s*\n\s*NNGraph::TensorNode\* src,)"
    r"\s*\n\s*const std::string& output_name,\s*\n\s*(Index ndim)"
)


def patch_nn_headers(text: str) -> str:
    text = text.replace(
        "NNGraph::TensorNode* forward(const std::string& output_name);",
        "NNGraph::TensorNode* forward();",
    )
    text = re.sub(
        _GEMM_HDR_RE,
        r"\1\n    \2",
        text,
        count=1,
    )
    text = re.sub(
        _TRANSPOSE_HDR_RE,
        r"\1\n    \2",
        text,
        count=1,
    )
    text = re.sub(r",\s*const std::string& output_name\)", ")", text)
    text = re.sub(r",\s*const std::string& output_name,", ",", text)
    return text


_FWD_DECL_RE = (
    r"NNGraph::TensorNode\* (NN\w+Op)::forward\("
    r"const std::string& output_name\)"
)


def process_cc(path: Path) -> None:
    text = path.read_text()
    orig = text
    text = text.replace(
        "NNGraph::TensorNode* NN", "NNGraph::TensorNode* NN",  # no-op anchor
    )
    text = re.sub(
        _FWD_DECL_RE,
        r"NNGraph::TensorNode* \1::forward()",
        text,
    )
    # drop last arg in op->forward(output_name) and similar
    text = re.sub(r"op->forward\(output_name\)", "op->forward()", text)
    text = re.sub(
        r"(\w+Op\* \w+ = [^;]+);\s*\n\s*NNGraph::TensorNode\* \w+ = "
        r"\w+->forward\(output_name\);",
        r"\1;\n    NNGraph::TensorNode* out = \2->forward();".replace(
            "\\2", "op"
        ),
        text,
    )
    # Simpler: forward(output_name) -> forward()
    text = re.sub(r"->forward\(output_name\)", "->forward()", text)
    # free fn: remove output_name param from definition
    text = re.sub(
        r",\s*const std::string& output_name\)\s*\n",
        ")\n",
        text,
    )
    # Forward bodies may still mention output_name; fix via patterns below.

    def repl_grad(m: re.Match[str]) -> str:
        slot = (
            f"graph->get_or_create_grad({m.group(1)}, "
            f"nn_grad_slot_name({m.group(1)}))"
        )
        return slot

    text = re.sub(
        r"graph->get_or_create_grad\((\w+), \1->name\(\) \+ \"_grad\"\)",
        repl_grad,
        text,
    )

    # Internal tensor names default to ""; callers label via ``set_name`` if
    # needed.
    # handled file-by-file below
    if text != orig:
        path.write_text(text)


def main() -> None:
    for path in sorted(NN_HH.glob("*.hh")):
        o = path.read_text()
        n = patch_nn_headers(o)
        if n != o:
            path.write_text(n)
            print("hh", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
