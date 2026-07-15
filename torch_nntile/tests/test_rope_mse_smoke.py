# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_rope_mse_smoke.py
# Smoke shapes for rope and mse_loss on device="nntile".

"""Forward-shape smokes for ``rope`` and ``mse_loss`` (skip without libnntile)."""

from __future__ import annotations

import pytest
import torch

import torch_nntile
from torch_nntile import _C
from torch_nntile.rope import rope, rope_sin_cos_from_position_ids
from torch_nntile.training import mse_loss

pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)


def test_rope_forward_shape_on_nntile():
    torch.manual_seed(0)
    batch, heads, seq, head_dim = 2, 4, 8, 16
    position_ids = (
        torch.arange(seq, dtype=torch.long).unsqueeze(0).expand(batch, seq)
    )
    sin, cos = rope_sin_cos_from_position_ids(position_ids, head_dim)
    x = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    sin_n = sin.to("nntile")
    cos_n = cos.to("nntile")
    x_n = x.to("nntile")
    y = rope(sin_n.unsqueeze(1), cos_n.unsqueeze(1), x_n)
    torch_nntile.compile_graph()
    torch_nntile.run()
    torch_nntile.wait()
    y_cpu = y.to("cpu")
    assert y_cpu.shape == (batch, heads, seq, head_dim)
    assert y_cpu.dtype == torch.float32


def test_mse_loss_forward_shape_on_nntile():
    torch.manual_seed(1)
    x = torch.randn(3, 5, 7, dtype=torch.float32, requires_grad=True)
    x_n = x.detach().to("nntile").requires_grad_(True)
    loss = mse_loss(x_n, scale=1.0 / x.numel())
    torch_nntile.compile_graph()
    torch_nntile.run()
    torch_nntile.wait()
    loss_cpu = loss.to("cpu")
    assert loss_cpu.ndim == 0
    assert loss_cpu.numel() == 1
    assert torch.isfinite(loss_cpu)
