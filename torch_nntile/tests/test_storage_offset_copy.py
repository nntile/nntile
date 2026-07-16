# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_storage_offset_copy.py
# CPU->nntile copy must honor storage_offset (B=1 causal label views).

from __future__ import annotations

import torch
import torch.nn.functional as F
from conftest import nntile_cpu
from torch_nntile.training import cross_entropy

import torch_nntile


def test_to_nntile_preserves_storage_offset_view():
    """batch[:, 1:] at B=1 is is_contiguous() with storage_offset != 0."""
    batch = torch.arange(32, dtype=torch.long).view(1, 32)
    labels_view = batch[:, 1:]
    assert labels_view.shape == (1, 31)
    assert labels_view.is_contiguous()
    assert labels_view.storage_offset() == 1

    roundtrip = nntile_cpu(labels_view.to("nntile"))
    assert torch.equal(roundtrip, labels_view)
    assert not torch.equal(roundtrip, batch[:, :31])


def test_cross_entropy_batch1_causal_label_view_matches_cpu():
    torch.manual_seed(0)
    batch = torch.randint(0, 256, (1, 32), dtype=torch.long)
    labels_view = batch[:, 1:]
    assert labels_view.storage_offset() == 1
    logits = torch.randn(1, 31, 256)

    ref = F.cross_entropy(logits.reshape(-1, 256), labels_view.reshape(-1))
    loss = cross_entropy(
        logits.to("nntile"),
        labels_view.to("nntile"),
        reduction="mean",
    )
    assert torch.allclose(nntile_cpu(loss), ref, rtol=1e-5, atol=1e-5)
