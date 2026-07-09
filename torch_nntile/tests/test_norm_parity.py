# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_norm_parity.py
# Parity tests for nntile 2-norm via TensorGraph (libnntile).

import torch
import pytest

import torch_nntile
from torch_nntile import _C
from conftest import nntile_cpu


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)


def _init_nntile() -> None:
    if not _C.is_context_initialized():
        torch_nntile.init_context(ncpu=2, ncuda=0, cpu_fallback=False)


@pytest.fixture(autouse=True)
def _nntile_context():
    _init_nntile()
    yield


def test_global_norm_matches_cpu():
    x_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x = x_cpu.to("nntile")

    y = torch.linalg.vector_norm(x, ord=2)
    y_cpu = torch.linalg.vector_norm(x_cpu, ord=2)

    assert y.device.type == "nntile"
    assert y.shape == ()
    assert torch.allclose(nntile_cpu(y), y_cpu, rtol=1e-5, atol=1e-5)


def test_axis_norm_dim0_matches_cpu():
    x_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x = x_cpu.to("nntile")

    y = torch.linalg.vector_norm(x, ord=2, dim=0)
    y_cpu = torch.linalg.vector_norm(x_cpu, ord=2, dim=0)

    assert torch.allclose(nntile_cpu(y), y_cpu, rtol=1e-5, atol=1e-5)


def test_axis_norm_dim1_matches_cpu():
    x_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x = x_cpu.to("nntile")

    y = torch.linalg.vector_norm(x, ord=2, dim=1)
    y_cpu = torch.linalg.vector_norm(x_cpu, ord=2, dim=1)

    assert torch.allclose(nntile_cpu(y), y_cpu, rtol=1e-5, atol=1e-5)


def test_axis_norm_keepdim_matches_cpu():
    x_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x = x_cpu.to("nntile")

    y = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
    y_cpu = torch.linalg.vector_norm(x_cpu, ord=2, dim=1, keepdim=True)

    assert y.shape == y_cpu.shape
    assert torch.allclose(nntile_cpu(y), y_cpu, rtol=1e-5, atol=1e-5)


def test_3d_axis_norm_matches_cpu():
    x_cpu = torch.randn(2, 3, 4, dtype=torch.float32)
    x = x_cpu.to("nntile")

    y = torch.linalg.vector_norm(x, ord=2, dim=1)
    y_cpu = torch.linalg.vector_norm(x_cpu, ord=2, dim=1)

    assert torch.allclose(nntile_cpu(y), y_cpu, rtol=1e-5, atol=1e-5)


def test_global_norm_keepdim_matches_cpu():
    x_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x = x_cpu.to("nntile")

    y = torch.linalg.vector_norm(x, ord=2, keepdim=True)
    y_cpu = torch.linalg.vector_norm(x_cpu, ord=2, keepdim=True)

    assert y.shape == y_cpu.shape
    assert torch.allclose(nntile_cpu(y), y_cpu, rtol=1e-5, atol=1e-5)


def test_vector_norm_out_matches_cpu():
    x_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out_cpu = torch.empty(2)
    y_cpu = torch.linalg.vector_norm(
        x_cpu,
        ord=2,
        dim=1,
        out=out_cpu,
    )

    x = x_cpu.to("nntile")
    out = torch.empty(2, device="nntile")
    y = torch.linalg.vector_norm(x, ord=2, dim=1, out=out)

    assert y is out
    assert torch.allclose(nntile_cpu(out), out_cpu, rtol=1e-5, atol=1e-5)
    assert torch.allclose(nntile_cpu(y), y_cpu, rtol=1e-5, atol=1e-5)
