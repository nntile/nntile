# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/classic_graph.py
"""Helpers: pending TensorGraph must be classic kernels, not TORCH_*."""

from __future__ import annotations

import torch_nntile


def pending_op_names() -> list[str]:
    return torch_nntile.pending_op_names()


# Autograd combines multiple incoming grads with aten add (residual
# fan-in, QKV from one activation, ...). That is the engine, not model
# compute; classic models still record GEMM / ADD / SILU / ... for the
# actual stack.
_AUTOGRAD_COMBINE_OPS = frozenset({"TORCH_BINARY"})


def assert_classic_graph() -> None:
    """Fail if the pending TensorGraph contains torch-native compute."""
    names = pending_op_names()
    torch_ops = [
        name
        for name in names
        if name.startswith("TORCH_") and name not in _AUTOGRAD_COMBINE_OPS
    ]
    if torch_ops:
        preview = ", ".join(torch_ops[:12])
        extra = "" if len(torch_ops) <= 12 else f" (+{len(torch_ops) - 12})"
        raise AssertionError(
            "pending TensorGraph has torch-native compute ops: "
            f"{preview}{extra}"
        )


def assert_torch_native_graph() -> None:
    """Fail if the pending TensorGraph has no torch-native ``TORCH_*`` ops."""
    names = pending_op_names()
    torch_ops = [name for name in names if name.startswith("TORCH_")]
    if not torch_ops:
        preview = ", ".join(names[:12]) if names else "<empty>"
        extra = "" if len(names) <= 12 else f" (+{len(names) - 12})"
        raise AssertionError(
            "pending TensorGraph has no torch-native TORCH_* ops: "
            f"{preview}{extra}"
        )
