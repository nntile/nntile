# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_cross_entropy_parity.py
# Cross-entropy parity: CPU PyTorch vs nntile tensor ops.

import torch
import pytest
import torch.nn.functional as F

import torch_nntile
from torch_nntile import _C
from torch_nntile.training import cross_entropy


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)


@pytest.fixture(scope="module", autouse=True)
def _nntile_context():
    if not torch_nntile.is_context_initialized():
        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
        )
    yield


def test_cross_entropy_forward_mean_matches_cpu():
    torch.manual_seed(0)
    batch, classes = 8, 5
    logits_cpu = torch.randn(batch, classes, dtype=torch.float32)
    target = torch.randint(0, classes, (batch,))

    loss_cpu = F.cross_entropy(logits_cpu, target, reduction="mean")
    logits_nnt = logits_cpu.detach().to("nntile")
    loss_nnt = cross_entropy(logits_nnt, target, reduction="mean")

    assert torch.allclose(
        loss_nnt.detach().cpu(),
        loss_cpu,
        rtol=1e-4,
        atol=1e-4,
    )


def test_cross_entropy_forward_sum_matches_cpu():
    torch.manual_seed(1)
    batch, classes = 4, 3
    logits_cpu = torch.randn(batch, classes, dtype=torch.float32)
    target = torch.randint(0, classes, (batch,))

    loss_cpu = F.cross_entropy(logits_cpu, target, reduction="sum")
    logits_nnt = logits_cpu.detach().to("nntile")
    loss_nnt = cross_entropy(logits_nnt, target, reduction="sum")

    assert torch.allclose(
        loss_nnt.detach().cpu(),
        loss_cpu,
        rtol=1e-4,
        atol=1e-4,
    )


def test_cross_entropy_backward_matches_cpu():
    torch.manual_seed(2)
    batch, classes = 6, 4
    logits_cpu = torch.randn(batch, classes, dtype=torch.float32, requires_grad=True)
    target = torch.randint(0, classes, (batch,))

    loss_cpu = F.cross_entropy(logits_cpu, target, reduction="mean")
    loss_cpu.backward()
    grad_cpu = logits_cpu.grad.detach().clone()

    logits_nnt = logits_cpu.detach().clone().to("nntile").requires_grad_(True)
    loss_nnt = cross_entropy(logits_nnt, target, reduction="mean")
    loss_nnt.backward()
    grad_nnt = logits_nnt.grad.cpu()

    assert torch.allclose(grad_nnt, grad_cpu, rtol=1e-4, atol=1e-4)


def test_cross_entropy_ignore_index_matches_cpu():
    torch.manual_seed(3)
    batch, classes = 5, 3
    logits_cpu = torch.randn(batch, classes, dtype=torch.float32)
    target = torch.tensor([0, 1, -100, 2, -100], dtype=torch.long)
    ignore_index = -100

    loss_cpu = F.cross_entropy(
        logits_cpu,
        target,
        reduction="mean",
        ignore_index=ignore_index,
    )
    logits_nnt = logits_cpu.detach().to("nntile")
    loss_nnt = cross_entropy(
        logits_nnt,
        target,
        reduction="mean",
        ignore_index=ignore_index,
    )

    assert torch.allclose(
        loss_nnt.detach().cpu(),
        loss_cpu,
        rtol=1e-4,
        atol=1e-4,
    )
