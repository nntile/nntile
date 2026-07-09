# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_axis_group_tiling.py
# Axis-group naming and tiling for torch_nntile graph recorder.

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from torch_nntile import _C

pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)

_PKG_ROOT = Path(__file__).resolve().parent.parent


def _run_subprocess(script: str) -> None:
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


def test_axis_group_tiling_add():
    _run_subprocess(
        """
        import torch
        import torch_nntile

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
        )
        torch_nntile.restrict_cpu()
        x = torch.randn(4, 8).to("nntile")
        y = torch.randn(4, 8).to("nntile")
        torch_nntile.set_axis_group_name(x, {0: "batch"})
        z = x + y
        torch_nntile.set_axis_group_tiling("batch", [1, 1, 2])
        torch_nntile.execute()
        assert torch.allclose(z.cpu(), (x.cpu() + y.cpu()))
        """
    )


def test_axis_group_tiling_invalid_sum_raises():
    _run_subprocess(
        """
        import pytest
        import torch
        import torch_nntile

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
        )
        x = torch.randn(4, 8).to("nntile")
        y = torch.randn(4, 8).to("nntile")
        torch_nntile.set_axis_group_name(x, {0: "batch"})
        _ = x + y
        torch_nntile.set_axis_group_tiling("batch", [1, 1, 1])
        with pytest.raises(RuntimeError, match="sum"):
            torch_nntile.execute()
        """
    )


def test_deep_relu_axis_groups_with_explicit_naming():
    _run_subprocess(
        """
        import torch
        import torch_nntile
        from torch_nntile.models import DeepReLU

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
        )
        model = DeepReLU.tiny().to("nntile")
        x = torch.randn(8, 128).to("nntile")
        logits = model(x)
        torch_nntile.set_axis_group_name(x, {0: "batch", 1: "features"})
        torch_nntile.set_axis_group_name(logits, {1: "classes"})
        for module in model.modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            w = module.weight
            names = {}
            if w.shape[0] == 256:
                names[0] = "hidden"
            if w.shape[1] == 256:
                names[1] = "hidden"
            if names:
                torch_nntile.set_axis_group_name(w, names)
        info = torch_nntile.format_axis_groups()
        assert "name='batch'" in info
        assert "name='features'" in info
        assert "name='classes'" in info
        assert "name='hidden'" in info
        torch_nntile.set_axis_group_tiling("batch", [4, 4])
        torch_nntile.set_axis_group_tiling("features", 64)
        torch_nntile.set_axis_group_tiling("hidden", 128)
        torch_nntile.execute()
        assert logits.shape == (8, 10)
        """
    )


def test_print_axis_groups_shows_pending_tiling():
    _run_subprocess(
        """
        import torch
        import torch_nntile

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
        )
        x = torch.randn(4, 8).to("nntile")
        y = torch.randn(4, 8).to("nntile")
        torch_nntile.set_axis_group_name(x, {0: "batch"})
        _ = x + y
        torch_nntile.set_axis_group_tiling("batch", [1, 1, 2])
        info = torch_nntile.format_axis_groups()
        assert "pending_tile=1,1,2" in info
        torch_nntile.execute()
        """
    )


def test_int64_label_ingress_with_batch_tiling():
    """INT64 scatter into a tiled logical must work (CE labels + --axis-tiling)."""
    _run_subprocess(
        """
        import torch
        import torch_nntile
        from torch_nntile.training import cross_entropy

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
        )
        torch_nntile.restrict_cpu()
        logits = torch.randn(8, 4).to("nntile")
        labels = torch.randint(0, 4, (8,), dtype=torch.long).to("nntile")
        torch_nntile.set_axis_group_name(logits, {0: "batch"})
        torch_nntile.set_axis_group_name(labels, {0: "batch"})
        loss = cross_entropy(logits, labels)
        torch_nntile.set_axis_group_tiling("batch", [4, 4])
        torch_nntile.compile_graph()
        torch_nntile.run()
        value = float(loss.to("cpu").item())
        assert value == value  # finite
        """
    )


def test_early_host_roundtrip_before_axis_tiling_raises():
    """``.cpu()`` before tiling seals untiled layouts; later tiling must fail."""
    _run_subprocess(
        """
        import pytest
        import torch
        import torch_nntile

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
        )
        torch_nntile.restrict_cpu()
        x = torch.randn(4, 8).to("nntile")
        y = torch.randn(4, 8).to("nntile")
        # Host round-trip compiles ingress scatter under the default
        # (untiled) layout before axis tiling is registered.
        _ = x.cpu()
        torch_nntile.set_axis_group_name(x, {0: "batch"})
        z = x + y
        torch_nntile.set_axis_group_tiling("batch", [1, 1, 2])
        with pytest.raises(RuntimeError, match="layout_fingerprint mismatch"):
            torch_nntile.execute()
        """
    )


def test_axis_tiling_without_early_host_roundtrip():
    """Axis tiling works when no host read seals the graph first."""
    _run_subprocess(
        """
        import torch
        import torch_nntile

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
        )
        torch_nntile.restrict_cpu()
        x = torch.randn(4, 8).to("nntile")
        y = torch.randn(4, 8).to("nntile")
        torch_nntile.set_axis_group_name(x, {0: "batch"})
        z = x + y
        torch_nntile.set_axis_group_tiling("batch", [1, 1, 2])
        torch_nntile.execute()
        assert z.shape == (4, 8)
        out = z.cpu()
        assert out.shape == (4, 8)
        """
    )
