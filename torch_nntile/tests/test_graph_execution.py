# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_graph_execution.py
# TensorGraph deferred execution for torch_nntile.

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


def test_deferred_until_execute():
    _run_graph_subprocess(
        """
        import torch
        import torch_nntile

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
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

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
        )
        torch_nntile.restrict_cpu()
        x = torch.randn(2, 3).to("nntile")
        w = torch.randn(4, 3).to("nntile")
        y = torch.nn.functional.relu(torch.nn.functional.linear(x, w, None))
        assert torch_nntile.has_pending_graph()
        y_cpu = y.cpu()
        assert y_cpu.shape == (2, 4)
        assert torch.isfinite(y_cpu).all()
        """
    )


def test_to_nntile_does_not_execute():
    _run_graph_subprocess(
        """
        import torch
        import torch_nntile

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
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
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
        )
        torch_nntile.restrict_cpu()
        model_graph = DeepReLU.tiny()
        model_graph.load_state_dict(model_cpu.state_dict())
        model_graph = model_graph.to("nntile")
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
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
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
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
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
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
        )
        torch_nntile.restrict_cpu()

        torch.manual_seed(0)
        batch, classes, features = 4, 3, 5
        w_cpu = torch.randn(classes, features)
        x_cpu = torch.randn(batch, features)
        target = torch.randint(0, classes, (batch,)).to("nntile")
        target_cpu = target.cpu()

        w_ref = w_cpu.clone().requires_grad_(True)
        logits_ref = x_cpu @ w_ref.t()
        loss_ref = torch.nn.functional.cross_entropy(logits_ref, target_cpu)
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


def test_train_full_batch_step():
    _run_graph_subprocess(
        """
        import math

        import torch
        import torch_nntile
        from torch_nntile.models import DeepReLU
        from torch_nntile.training import clone_model_weights, train_full_batch_step

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
        )
        torch_nntile.restrict_cpu()

        model_cpu = DeepReLU.tiny()
        model_cpu.init_kaiming_uniform_(seed=42)
        model = DeepReLU.tiny()
        model.load_state_dict(model_cpu.state_dict())
        model = model.to("nntile")
        before = clone_model_weights(model)
        x = torch.randn(8, model.input_dim).to("nntile")
        y = torch.randint(0, model.output_dim, (8,)).to("nntile")
        loss = train_full_batch_step(model, x, y, learning_rate=0.1)
        assert math.isfinite(loss)
        after = clone_model_weights(model)
        max_delta = max((before[k] - after[k]).abs().max().item() for k in before)
        assert max_delta > 0.0
        """
    )


def test_train_full_batch_step_multi_epoch():
    """compile_graph() clears pending nodes; epoch 2 must recreate data nodes."""
    _run_graph_subprocess(
        """
        import math

        import torch
        import torch_nntile
        from torch_nntile.models import DeepReLU
        from torch_nntile.training import (
            clone_model_weights,
            max_weight_delta,
            train_full_batch_step,
        )

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
        )
        torch_nntile.restrict_cpu()

        model_cpu = DeepReLU.tiny()
        model_cpu.init_kaiming_uniform_(seed=42)
        model = DeepReLU.tiny()
        model.load_state_dict(model_cpu.state_dict())
        model = model.to("nntile")
        x = torch.randn(8, model.input_dim).to("nntile")
        y = torch.randint(0, model.output_dim, (8,)).to("nntile")
        w0 = clone_model_weights(model)
        loss1 = train_full_batch_step(model, x, y, learning_rate=0.1)
        loss2 = train_full_batch_step(model, x, y, learning_rate=0.1)
        loss3 = train_full_batch_step(model, x, y, learning_rate=0.1)
        assert all(math.isfinite(v) for v in (loss1, loss2, loss3))
        assert loss2 < loss1
        assert loss3 < loss2
        """
    )


def test_graph_nntile_loss_backward_without_scalar_read():
    _run_graph_subprocess(
        """
        import torch
        import torch_nntile
        from torch_nntile.training import cross_entropy

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
        )
        torch_nntile.restrict_cpu()

        logits_cpu = torch.randn(8, 5)
        target = torch.randint(0, 5, (8,)).to("nntile")
        target_cpu = target.cpu()
        logits_nnt = logits_cpu.to("nntile").requires_grad_(True)
        loss = cross_entropy(logits_nnt, target, reduction="mean")
        assert loss.device.type == "nntile"
        assert torch_nntile.has_pending_graph()
        loss.backward()
        assert torch_nntile.has_pending_graph()
        torch_nntile.execute()
        ref = torch.nn.functional.cross_entropy(
            logits_cpu, target_cpu, reduction="mean"
        )
        assert torch.allclose(loss.detach().cpu(), ref, rtol=1e-4, atol=1e-4)
        """
    )


def test_mm_view_add_ndim():
    """mm output viewed to higher rank must match bias ndim."""
    _run_graph_subprocess(
        """
        import torch
        import torch_nntile

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
        )
        torch_nntile.restrict_cpu()

        x2d = torch.randn(4, 8).to("nntile")
        w = torch.randn(8, 32).to("nntile")
        bias = torch.randn(1, 4, 16, 2).to("nntile")

        proj = torch.mm(x2d, w)
        proj4 = proj.view(1, 4, 16, 2)
        out = proj4 + bias
        assert torch_nntile.has_pending_graph()
        torch_nntile.compile_graph()
        torch_nntile.run()
        assert out.shape == (1, 4, 16, 2)
        assert out.device.type == "nntile"
        """
    )


def test_intermediate_output_mark_cleared_when_python_ref_dropped():
    _run_graph_subprocess(
        """
        import gc
        import torch
        import torch_nntile

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
        )
        torch_nntile.restrict_cpu()
        a = torch.tensor([1.0, 2.0]).to("nntile")
        b = torch.tensor([3.0, 4.0]).to("nntile")
        c = torch.tensor([0.5, 0.5]).to("nntile")
        t = a + b
        d = t + c
        del t
        gc.collect()
        torch_nntile.compile_graph()
        torch_nntile.run()
        torch_nntile.wait()
        assert torch.allclose(d.cpu(), torch.tensor([4.5, 6.5]))
        """
    )


def test_intermediate_output_mark_kept_while_python_ref_alive():
    _run_graph_subprocess(
        """
        import torch
        import torch_nntile

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
        )
        torch_nntile.restrict_cpu()
        a = torch.tensor([1.0, 2.0]).to("nntile")
        b = torch.tensor([3.0, 4.0]).to("nntile")
        c = torch.tensor([0.5, 0.5]).to("nntile")
        t = a + b
        d = t + c
        torch_nntile.compile_graph()
        torch_nntile.run()
        torch_nntile.wait()
        assert torch.allclose(t.cpu(), torch.tensor([4.0, 6.0]))
        assert torch.allclose(d.cpu(), torch.tensor([4.5, 6.5]))
        """
    )


def test_clean_exit_with_live_nntile_tensors_after_atexit():
    """Bindings must not UAF TensorNodes destroyed by atexit shutdown."""
    _run_graph_subprocess(
        """
        import torch
        import torch_nntile

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
        )
        torch_nntile.restrict_cpu()
        # Keep live nntile tensors across process exit so atexit
        # shutdown_context() tears down TensorGraph before Python GC
        # destroys the TensorImpl / NodeRef shells.
        x = torch.randn(4, 4).to("nntile")
        y = torch.nn.functional.relu(x)
        torch_nntile.compile_graph()
        torch_nntile.run()
        # Intentionally leave x/y alive until interpreter teardown.
        assert x.device.type == "nntile"
        assert y.device.type == "nntile"
        """
    )
