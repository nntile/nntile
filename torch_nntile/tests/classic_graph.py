# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/classic_graph.py
"""Helpers: pending TensorGraph must be classic kernels, not TORCH_*."""

from __future__ import annotations

import torch_nntile


def pending_op_names() -> list[str]:
    return torch_nntile.pending_op_names()


def assert_classic_graph() -> None:
    """Fail if the pending TensorGraph contains torch-native compute."""
    names = pending_op_names()
    torch_ops = [
        name
        for name in names
        if name.startswith("TORCH_")
    ]
    if torch_ops:
        preview = ", ".join(torch_ops[:12])
        extra = "" if len(torch_ops) <= 12 else f" (+{len(torch_ops) - 12})"
        raise AssertionError(
            "pending TensorGraph has torch-native compute ops: "
            f"{preview}{extra}"
        )
