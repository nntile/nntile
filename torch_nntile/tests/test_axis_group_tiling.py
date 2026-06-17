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


def test_axis_group_tiling_add_graph_mode():
    _run_subprocess(
        """
        import torch
        import torch_nntile

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False, runtime_mode="graph"
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
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False, runtime_mode="graph"
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


def test_deep_relu_names_axis_groups_in_forward():
    _run_subprocess(
        """
        import torch
        import torch_nntile
        from torch_nntile.models import DeepReLU

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False, runtime_mode="graph"
        )
        model = DeepReLU.tiny().to("nntile")
        x = torch.randn(8, 128).to("nntile")
        logits = model(x)
        torch_nntile.set_axis_group_tiling("batch", [4, 4])
        torch_nntile.set_axis_group_tiling("features", 64)
        torch_nntile.set_axis_group_tiling("classes", 5)
        torch_nntile.execute()
        assert logits.shape == (8, 10)
        """
    )
