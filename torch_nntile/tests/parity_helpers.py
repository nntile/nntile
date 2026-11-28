# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/parity_helpers.py
# Shared helpers for CPU / HF / nntile model submodule parity tests.
# @version 2.0.0

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from conftest import nntile_cpu
from torch import Tensor


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


def contiguous_to_nntile(tensor: Tensor) -> Tensor:
    """Ensure CPU contiguous layout before ``.to('nntile')``."""
    return tensor.detach().cpu().contiguous().to("nntile")
