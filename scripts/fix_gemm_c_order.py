#!/usr/bin/env python3
"""Post-fix GEMM test shapes after blind shape reversal."""

from __future__ import annotations

import re
from pathlib import Path

GEMM_FILES = list(Path("nntile/tests").rglob("gemm.cc"))


def fix_text(text: str) -> str:
    # B operand: {N, K} -> {K, N} (2D variable shapes)
    text = text.replace("g.tensor({N, K}", "g.tensor({K, N}")
    text = text.replace("graph.data({6, 5})", "graph.data({5, 6})")
    # B batched: {B, N, K} -> {B, K, N}
    text = text.replace("g.tensor({B, N, K}", "g.tensor({B, K, N}")
    # 4D B: {N2, N1, K2, K1} -> {K1, K2, N1, N2}
    text = text.replace(
        "g.tensor({N2, N1, K2, K1}", "g.tensor({K1, K2, N1, N2}"
    )
    # A grad 2D
    text = text.replace(
        "(std::vector<Index>{M, K})", "(std::vector<Index>{K, M})"
    )
    text = text.replace(
        "REQUIRE(a->grad()->shape() == (std::vector<Index>{M, K, B}));",
        "REQUIRE(a->grad()->shape() == (std::vector<Index>{B, K, M}));",
    )
    text = text.replace(
        "REQUIRE(b->grad()->shape() == (std::vector<Index>{K, N, B}));",
        "REQUIRE(b->grad()->shape() == (std::vector<Index>{B, K, N}));",
    )
    text = text.replace(
        "REQUIRE(c->shape() == (std::vector<Index>{M, N, B}));",
        "REQUIRE(c->shape() == (std::vector<Index>{B, M, N}));",
    )
    # 4D grad shapes
    text = text.replace(
        "(std::vector<Index>{M1, M2, K1, K2})",
        "(std::vector<Index>{K2, K1, M2, M1})",
    )
    text = text.replace(
        "(std::vector<Index>{M1, K1, K2})",
        "(std::vector<Index>{K2, K1, M1})",
    )
    text = text.replace(
        "REQUIRE(c->shape() == (std::vector<Index>{M1, M2, N1, N2}));",
        "REQUIRE(c->shape() == (std::vector<Index>{N2, N1, M2, M1}));",
    )
    text = text.replace(
        "REQUIRE(c->shape() == (std::vector<Index>{M1, N1, N2}));",
        "REQUIRE(c->shape() == (std::vector<Index>{N2, N1, M1}));",
    )
    # GENERATE expected shapes for multi-dim forward/backward
    text = text.replace(
        "std::vector<Index>{2, 3, 3, 5}",
        "std::vector<Index>{5, 3, 3, 2}",
    )
    text = text.replace(
        "std::vector<Index>{2, 3, 3}",
        "std::vector<Index>{3, 3, 2}",
    )
    text = text.replace(
        "std::vector<Index>{2, 5, 6}",
        "std::vector<Index>{6, 5, 2}",
    )
    text = text.replace(
        "std::vector<Index>{4, 2, 3, 5}",
        "std::vector<Index>{5, 3, 2, 4}",
    )
    text = text.replace(
        "std::vector<Index>{4, 3, 3}",
        "std::vector<Index>{3, 3, 4}",
    )
    text = text.replace(
        "std::vector<Index>{3, 4, 5, 6}",
        "std::vector<Index>{6, 5, 4, 3}",
    )
    # PyTorch from_blob shapes for 2D gemm (after colmajor removal)
    text = re.sub(
        r"torch::from_blob\(a_data\.data\(\),\s*\{M, K\}",
        "torch::from_blob(a_data.data(), {K, M}",
        text,
    )
    text = re.sub(
        r"torch::from_blob\(b_data\.data\(\),\s*\{K, N\}",
        "torch::from_blob(b_data.data(), {K, N}",
        text,
    )
    text = re.sub(
        r"torch::from_blob\(a_data\.data\(\),\s*\{M1, M2, K1, K2\}",
        "torch::from_blob(a_data.data(), {K2, K1, M2, M1}",
        text,
    )
    text = re.sub(
        r"torch::from_blob\(b_data\.data\(\),\s*\{K1, K2, N1, N2\}",
        "torch::from_blob(b_data.data(), {K1, K2, N1, N2}",
        text,
    )
    text = re.sub(
        r"torch::from_blob\(a_data\.data\(\),\s*\{M, K, B\}",
        "torch::from_blob(a_data.data(), {B, K, M}",
        text,
    )
    text = re.sub(
        r"torch::from_blob\(b_data\.data\(\),\s*\{K, N, B\}",
        "torch::from_blob(b_data.data(), {B, K, N}",
        text,
    )
    text = re.sub(
        r"\.reshape\(\{M1, M2, N1, N2\}\)",
        ".reshape({N2, N1, M2, M1})",
        text,
    )
    text = re.sub(
        r"\.reshape\(\{M1, M2, K1, K2\}\)",
        ".reshape({K2, K1, M2, M1})",
        text,
    )
    text = re.sub(
        r"\.reshape\(\{K1, K2, N1, N2\}\)",
        ".reshape({K1, K2, N1, N2})",
        text,
    )
    text = re.sub(
        r"torch::full\(\{M, N\}",
        "torch::full({M, N}",
        text,
    )
    return text


def main() -> None:
    for path in GEMM_FILES:
        text = path.read_text()
        new = fix_text(text)
        if new != text:
            path.write_text(new)
            print("fixed", path)


if __name__ == "__main__":
    main()
