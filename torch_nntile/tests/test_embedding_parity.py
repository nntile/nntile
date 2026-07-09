# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_embedding_parity.py
# Embedding parity: CPU PyTorch vs nntile tensor ops.

import subprocess
import sys
import textwrap
from pathlib import Path

import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F

import torch_nntile
from torch_nntile import _C
from conftest import nntile_cpu


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)

_PKG_ROOT = Path(__file__).resolve().parent.parent


def _run_graph_subprocess(script: str) -> None:
    env = dict(**__import__("os").environ)
    repo = Path(__file__).resolve().parents[2]
    build_lib = repo / "build" / "nntile"
    starpu_lib = "/opt/starpu/lib"
    ld = env.get("LD_LIBRARY_PATH", "")
    for part in (str(build_lib), starpu_lib):
        if part not in ld.split(":"):
            ld = f"{part}:{ld}" if ld else part
    env["LD_LIBRARY_PATH"] = ld
    env["PYTHONPATH"] = f"{_PKG_ROOT}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"subprocess failed ({proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


@pytest.mark.parametrize("index_shape", [(8,), (4, 5), (2, 3, 4)])
def test_embedding_forward_matches_cpu(index_shape):
    torch.manual_seed(0)
    num_embeddings, embed_dim = 16, 7
    weight_cpu = torch.randn(num_embeddings, embed_dim, dtype=torch.float32)
    indices = torch.randint(0, num_embeddings, index_shape)

    out_cpu = F.embedding(indices, weight_cpu)
    weight_nnt = weight_cpu.detach().to("nntile")
    out_nnt = F.embedding(indices.to("nntile"), weight_nnt)

    assert torch.allclose(nntile_cpu(out_nnt.detach()), out_cpu, rtol=1e-5, atol=1e-5)


def test_embedding_backward_matches_cpu():
    torch.manual_seed(1)
    num_embeddings, embed_dim = 12, 6
    indices = torch.randint(0, num_embeddings, (5, 4))

    weight_cpu = nn.Embedding(num_embeddings, embed_dim)
    torch.nn.init.normal_(weight_cpu.weight, mean=0.0, std=0.1)
    weight_cpu.weight.requires_grad_(True)

    out_cpu = weight_cpu(indices)
    out_cpu.sum().backward()
    grad_cpu = weight_cpu.weight.grad.detach().clone()

    weight_nnt = nn.Embedding(num_embeddings, embed_dim)
    weight_nnt.load_state_dict(weight_cpu.state_dict())
    weight_nnt = weight_nnt.to("nntile")
    weight_nnt.weight.requires_grad_(True)

    out_nnt = weight_nnt(indices.to("nntile"))
    grad_out = torch.ones_like(out_nnt)
    torch.autograd.backward([out_nnt], [grad_out])
    grad_nnt = nntile_cpu(weight_nnt.weight.grad)

    assert torch.allclose(grad_nnt, grad_cpu, rtol=1e-5, atol=1e-5)


def test_embedding_duplicate_indices():
    torch.manual_seed(2)
    num_embeddings, embed_dim = 6, 4
    indices = torch.tensor([0, 1, 0, 2, 1, 0])

    weight_cpu = nn.Embedding(num_embeddings, embed_dim)
    torch.nn.init.normal_(weight_cpu.weight, mean=0.0, std=0.2)
    weight_cpu.weight.requires_grad_(True)

    out_cpu = weight_cpu(indices)
    out_cpu.sum().backward()
    grad_cpu = weight_cpu.weight.grad.detach().clone()

    weight_nnt = nn.Embedding(num_embeddings, embed_dim)
    weight_nnt.load_state_dict(weight_cpu.state_dict())
    weight_nnt = weight_nnt.to("nntile")
    weight_nnt.weight.requires_grad_(True)

    out_nnt = weight_nnt(indices.to("nntile"))
    grad_out = torch.ones_like(out_nnt)
    torch.autograd.backward([out_nnt], [grad_out])
    grad_nnt = nntile_cpu(weight_nnt.weight.grad)

    assert torch.allclose(grad_nnt, grad_cpu, rtol=1e-5, atol=1e-5)


def test_embedding_rejects_cpu_indices():
    torch.manual_seed(3)
    num_embeddings, embed_dim = 10, 5
    indices = torch.randint(0, num_embeddings, (3, 3))
    weight_nnt = torch.randn(num_embeddings, embed_dim, dtype=torch.float32).to(
        "nntile"
    )

    assert indices.device.type == "cpu"
    with pytest.raises(RuntimeError, match="indices must be on device nntile"):
        F.embedding(indices, weight_nnt)


def test_embedding_deferred_until_compile():
    _run_graph_subprocess(
        """
        import torch
        import torch.nn.functional as F
        import torch_nntile

        torch.manual_seed(4)
        torch_nntile.init_context(
            ncpu=1, ncuda=0, cpu_fallback=False
        )
        torch_nntile.restrict_cpu()

        num_embeddings, embed_dim = 8, 4
        indices = torch.randint(0, num_embeddings, (2, 3)).to("nntile")
        weight_cpu = torch.randn(num_embeddings, embed_dim, dtype=torch.float32)
        out_cpu = F.embedding(indices.cpu(), weight_cpu)

        weight_nnt = weight_cpu.detach().to("nntile")
        out_nnt = F.embedding(indices, weight_nnt)
        assert torch_nntile.has_pending_graph()
        torch_nntile.compile_graph()
        torch_nntile.run()
        assert torch.allclose(
            out_nnt.detach().cpu(), out_cpu, rtol=1e-5, atol=1e-5
        )
        """
    )
