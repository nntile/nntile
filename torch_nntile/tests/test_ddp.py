# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_ddp.py
# Product DDP contract: named batch axis + torch_nntile.ddp().

from __future__ import annotations

import subprocess
import sys
import textwrap

from conftest import subprocess_environ


def _run_subprocess(script: str) -> None:
    env = subprocess_environ()
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


def test_ddp_linear_ce_matches_cpu():
    _run_subprocess(
        """
        import torch
        import torch_nntile
        from torch_nntile.nn.module import Linear
        from torch_nntile.training import SGD, cross_entropy

        torch.manual_seed(0)
        x = torch.randn(4, 8)
        y = torch.randint(0, 3, (4,), dtype=torch.long)
        cpu_layer = torch.nn.Linear(8, 3, bias=False)
        logits_cpu = cpu_layer(x)
        loss_cpu = torch.nn.functional.cross_entropy(logits_cpu, y)
        loss_cpu.backward()
        with torch.no_grad():
            w_cpu = cpu_layer.weight - 0.1 * cpu_layer.weight.grad

        torch_nntile.init_context(
            ncpu=2, ncuda=0, verbose=0, cpu_fallback=False
        )
        torch_nntile.restrict_cpu()
        layer = Linear(8, 3, bias=False)
        with torch.no_grad():
            layer.load_state_dict(cpu_layer.state_dict())
            layer = layer.to("nntile")
            x_n = x.to("nntile")
            y_n = y.to("nntile")
        logits = layer(x_n)
        loss = cross_entropy(logits, y_n)
        loss.backward()
        opt = SGD([layer.weight], lr=0.1)
        opt.step()
        torch_nntile.set_axis_group_name(x_n, {0: "batch"})
        torch_nntile.set_axis_group_name(y_n, {0: "batch"})
        torch_nntile.ddp()
        info = torch_nntile.format_axis_groups()
        assert "DDP axis='batch'" in info
        assert "replicas=2" in info
        step_loss = loss.detach()
        del logits
        del loss
        opt.zero_grad(set_to_none=True)
        torch_nntile.compile_graph()
        torch_nntile.run()
        loss_n = float(step_loss.to("cpu").item())
        w_n = layer.weight.detach().cpu()
        assert abs(loss_n - float(loss_cpu.item())) < 1e-4
        assert torch.allclose(w_n, w_cpu, rtol=1e-4, atol=1e-4)
        """
    )


def test_ddp_rejects_second_axis():
    _run_subprocess(
        """
        import pytest
        import torch
        import torch_nntile

        torch_nntile.init_context(
            ncpu=2, ncuda=0, verbose=0, cpu_fallback=False
        )
        torch_nntile.ddp("batch")
        torch_nntile.ddp("batch")
        with pytest.raises(RuntimeError, match="already enabled"):
            torch_nntile.ddp("hidden")
        """
    )


def test_ddp_llama_train_step():
    _run_subprocess(
        """
        import torch
        import torch_nntile
        from torch_nntile.nn.model.llama import LlamaCausal, LlamaConfig
        from torch_nntile.training import AdamW, cross_entropy

        torch_nntile.init_context(
            ncpu=2, ncuda=0, verbose=0, cpu_fallback=False
        )
        torch_nntile.restrict_cpu()
        cfg = LlamaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=16,
        )
        model = LlamaCausal(cfg).float().train().to("nntile")
        g = torch.Generator().manual_seed(0)
        packed = torch.randint(
            0, 128, (4, 9), dtype=torch.long, generator=g
        )
        ids = packed[:, :-1].contiguous().to("nntile")
        labels = packed[:, 1:].contiguous().to("nntile")
        torch_nntile.set_axis_group_name(ids, {0: "batch"})
        torch_nntile.set_axis_group_name(labels, {0: "batch"})
        torch_nntile.ddp()
        torch_nntile.compile_graph()
        torch_nntile.run()
        torch_nntile.wait()
        for p in model.parameters():
            p.requires_grad_(True)
        opt = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3,
        )
        opt.zero_grad(set_to_none=True)
        logits = model(ids)
        loss = cross_entropy(logits, labels, reduction="mean")
        loss.backward()
        opt.step()
        names = torch_nntile.pending_op_names()
        torch_ops = [n for n in names if n.startswith("TORCH_")]
        assert not torch_ops, torch_ops
        sin, cos = model.model._rope_cache[8]
        half = cfg.hidden_size // cfg.num_attention_heads // 2
        assert tuple(sin.shape) == (8, half)
        assert tuple(cos.shape) == (8, half)
        torch_nntile.set_axis_group_name(logits, {0: "batch"})
        torch_nntile.ddp()
        info = torch_nntile.format_axis_groups()
        assert "DDP axis='batch'" in info
        assert "replicas=2" in info
        step_loss = loss.detach()
        del logits
        del loss
        opt.zero_grad(set_to_none=True)
        torch_nntile.compile_graph()
        torch_nntile.run()
        torch_nntile.wait()
        val = float(step_loss.to("cpu").item())
        assert val == val
        assert val > 0.0
        """
    )


def test_ddp_torch_add_raises():
    _run_subprocess(
        """
        import pytest
        import torch
        import torch_nntile

        torch_nntile.init_context(
            ncpu=2, ncuda=0, verbose=0, cpu_fallback=False
        )
        torch_nntile.restrict_cpu()
        x = torch.randn(4, 8).to("nntile")
        y = torch.randn(4, 8).to("nntile")
        torch_nntile.set_axis_group_name(x, {0: "batch"})
        _ = x + y
        torch_nntile.ddp()
        with pytest.raises(RuntimeError, match="TORCH_"):
            torch_nntile.execute()
        """
    )
