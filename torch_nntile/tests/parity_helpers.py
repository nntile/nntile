# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/parity_helpers.py
# Shared helpers for CPU vs nntile parity tests.

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch import Tensor

from conftest import nntile_cpu


def assert_close(
    got: Tensor,
    ref: Tensor,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    """Assert ``got`` (possibly on nntile) matches ``ref`` on CPU."""
    torch.testing.assert_close(
        nntile_cpu(got),
        ref.detach().cpu(),
        rtol=rtol,
        atol=atol,
    )


def clone_to_nntile(module: nn.Module) -> nn.Module:
    """Deep-copy ``module`` and move parameters/buffers to ``device=nntile``."""
    cloned = copy.deepcopy(module)
    with torch.no_grad():
        return cloned.to("nntile")


def rel_frobenius(a: Tensor, b: Tensor) -> float:
    """Relative Frobenius distance ``||a-b||_F / (||b||_F + eps)``."""
    a_cpu = nntile_cpu(a).float()
    b_cpu = nntile_cpu(b).float()
    diff = torch.linalg.norm(a_cpu - b_cpu)
    denom = torch.linalg.norm(b_cpu) + 1e-12
    return float((diff / denom).item())


def contiguous_to_nntile(tensor: Tensor) -> Tensor:
    """Ensure CPU contiguous layout before ``.to('nntile')``."""
    return tensor.detach().cpu().contiguous().to("nntile")
