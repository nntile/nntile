# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_graph_execution.py
# Graph (non-eager) runtime mode for torch_nntile.

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from torch_nntile import _C

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


def test_graph_mode_deferred_until_execute():
    _run_graph_subprocess(
        """
        import torch
        import torch_nntile

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False, runtime_mode="graph"
        )
        torch_nntile.restrict_cpu()
        x = torch.randn(2, 3).to("nntile")
        w = torch.randn(4, 3).to("nntile")
        h = torch.nn.functional.linear(x, w, None)
        assert torch_nntile.has_pending_graph()
        y = torch.nn.functional.relu(h)
        assert torch_nntile.has_pending_graph()
        torch_nntile.execute()
        assert not torch_nntile.has_pending_graph()
        """
    )


def test_cpu_copy_requires_execute():
    _run_graph_subprocess(
        """
        import torch
        import torch_nntile
        import pytest

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False, runtime_mode="graph"
        )
        torch_nntile.restrict_cpu()
        x = torch.randn(2, 3).to("nntile")
        w = torch.randn(4, 3).to("nntile")
        y = torch.nn.functional.relu(torch.nn.functional.linear(x, w, None))
        assert torch_nntile.has_pending_graph()
        with pytest.raises(RuntimeError, match="torch_nntile.execute"):
            y.cpu()
        torch_nntile.execute()
        y_cpu = y.cpu()
        assert y_cpu.shape == (2, 4)
        """
    )


def test_to_nntile_does_not_execute():
    _run_graph_subprocess(
        """
        import torch
        import torch_nntile

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False, runtime_mode="graph"
        )
        torch_nntile.restrict_cpu()
        x = x_cpu = torch.randn(2, 3)
        x = x_cpu.to("nntile")
        w = torch.randn(4, 3).to("nntile")
        _ = torch.nn.functional.linear(x, w, None)
        assert torch_nntile.has_pending_graph()
        """
    )


def test_graph_forward_matches_cpu():
    _run_graph_subprocess(
        """
        import torch
        import torch_nntile
        from torch_nntile.models import DeepReLU

        torch.manual_seed(0)
        model_cpu = DeepReLU.tiny()
        model_cpu.init_kaiming_uniform_(seed=42)
        x_cpu = torch.randn(32, model_cpu.input_dim)

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False, runtime_mode="graph"
        )
        torch_nntile.restrict_cpu()
        model_graph = DeepReLU.tiny().to("nntile")
        model_graph.load_state_dict(model_cpu.state_dict())
        with torch.no_grad():
            y_graph = model_graph(x_cpu.to("nntile"))
            assert torch_nntile.has_pending_graph()
            torch_nntile.execute()
            y_graph = y_graph.cpu()

        y_ref = model_cpu(x_cpu)
        assert torch.allclose(y_graph, y_ref, rtol=1e-4, atol=1e-4)
        """
    )


def test_graph_backward_without_mid_execute():
    _run_graph_subprocess(
        """
        import torch
        import torch_nntile

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False, runtime_mode="graph"
        )
        torch_nntile.restrict_cpu()

        x_cpu = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]], requires_grad=True)
        w_cpu = torch.tensor(
            [[0.25, -0.5, 1.0], [2.0, 0.0, -1.0]], requires_grad=True
        )
        y_cpu = torch.nn.functional.relu(x_cpu @ w_cpu.t())
        y_cpu.backward(torch.ones_like(y_cpu))

        x_nnt = x_cpu.detach().to("nntile").requires_grad_(True)
        w_nnt = w_cpu.detach().to("nntile").requires_grad_(True)
        y_nnt = torch.nn.functional.relu(
            torch.nn.functional.linear(x_nnt, w_nnt, None)
        )
        assert torch_nntile.has_pending_graph()
        y_nnt.backward(torch.ones(y_nnt.shape, device="cpu").to("nntile"))
        assert torch_nntile.has_pending_graph()
        torch_nntile.execute()
        gx = x_nnt.grad.cpu()
        gw = w_nnt.grad.cpu()
        assert torch.allclose(gx, x_cpu.grad, rtol=1e-4, atol=1e-4)
        assert torch.allclose(gw, w_cpu.grad, rtol=1e-4, atol=1e-4)
        """
    )


def test_execute_idempotent_on_empty():
    _run_graph_subprocess(
        """
        import torch_nntile

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False, runtime_mode="graph"
        )
        torch_nntile.restrict_cpu()
        assert not torch_nntile.has_pending_graph()
        torch_nntile.execute()
        torch_nntile.execute()
        """
    )


def test_graph_cross_entropy_backward_and_sgd():
    _run_graph_subprocess(
        """
        import torch
        import torch_nntile
        from torch_nntile.training import SGD, cross_entropy

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False, runtime_mode="graph"
        )
        torch_nntile.restrict_cpu()

        torch.manual_seed(0)
        batch, classes, features = 4, 3, 5
        w_cpu = torch.randn(classes, features)
        x_cpu = torch.randn(batch, features)
        target = torch.randint(0, classes, (batch,))

        w_ref = w_cpu.clone().requires_grad_(True)
        logits_ref = x_cpu @ w_ref.t()
        loss_ref = torch.nn.functional.cross_entropy(logits_ref, target)
        loss_ref.backward()
        torch.optim.SGD([w_ref], lr=0.1).step()
        w_after_ref = w_ref.detach().clone()

        w_nnt = w_cpu.detach().to("nntile").requires_grad_(True)
        x_nnt = x_cpu.to("nntile")
        logits_nnt = torch.nn.functional.linear(x_nnt, w_nnt, None)
        loss_nnt = cross_entropy(logits_nnt, target, reduction="mean")
        assert torch_nntile.has_pending_graph()
        loss_nnt.backward()
        assert torch_nntile.has_pending_graph()
        SGD([w_nnt], lr=0.1).step()
        assert torch_nntile.has_pending_graph()
        assert loss_nnt.device.type == "nntile"
        torch_nntile.execute()
        assert not torch_nntile.has_pending_graph()
        assert torch.allclose(loss_nnt.detach().cpu(), loss_ref, rtol=1e-4, atol=1e-4)
        assert torch.allclose(w_nnt.detach().cpu(), w_after_ref, rtol=1e-4, atol=1e-4)
        """
    )


def test_train_full_batch_step_graph_mode():
    _run_graph_subprocess(
        """
        import math

        import torch
        import torch_nntile
        from torch_nntile.models import DeepReLU
        from torch_nntile.training import clone_model_weights, train_full_batch_step

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False, runtime_mode="graph"
        )
        torch_nntile.restrict_cpu()

        model_cpu = DeepReLU.tiny()
        model_cpu.init_kaiming_uniform_(seed=42)
        model = DeepReLU.tiny().to("nntile")
        model.load_state_dict(model_cpu.state_dict())
        before = clone_model_weights(model)
        x = torch.randn(8, model.input_dim).to("nntile")
        y = torch.randint(0, model.output_dim, (8,))
        loss = train_full_batch_step(model, x, y, learning_rate=0.1)
        assert math.isfinite(loss)
        after = clone_model_weights(model)
        max_delta = max((before[k] - after[k]).abs().max().item() for k in before)
        assert max_delta > 0.0
        """
    )


def test_graph_nntile_loss_backward_without_scalar_read():
    _run_graph_subprocess(
        """
        import torch
        import torch_nntile
        from torch_nntile.training import cross_entropy

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False, runtime_mode="graph"
        )
        torch_nntile.restrict_cpu()

        logits_cpu = torch.randn(8, 5)
        target = torch.randint(0, 5, (8,))
        logits_nnt = logits_cpu.to("nntile").requires_grad_(True)
        loss = cross_entropy(logits_nnt, target, reduction="mean")
        assert loss.device.type == "nntile"
        assert torch_nntile.has_pending_graph()
        loss.backward()
        assert torch_nntile.has_pending_graph()
        torch_nntile.execute()
        ref = torch.nn.functional.cross_entropy(logits_cpu, target, reduction="mean")
        assert torch.allclose(loss.detach().cpu(), ref, rtol=1e-4, atol=1e-4)
        """
    )


def test_eager_mode_runs_immediately():
    _run_graph_subprocess(
        """
        import torch
        import torch_nntile

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False, runtime_mode="eager"
        )
        torch_nntile.restrict_cpu()
        a = torch.tensor([1.0, 2.0], device="nntile")
        b = torch.tensor([3.0, 4.0], device="nntile")
        z = a + b
        assert not torch_nntile.has_pending_graph()
        assert torch.allclose(z.cpu(), torch.tensor([4.0, 6.0]))
        """
    )
