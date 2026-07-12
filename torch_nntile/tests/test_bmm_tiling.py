# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_bmm_tiling.py

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


def _run_subprocess(script: str) -> None:
    env = dict(**__import__("os").environ)
    # Bench scripts may leave STARPU_DISABLE_KERNELS=1 in the parent env.
    env.pop("STARPU_DISABLE_KERNELS", None)
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


def test_bmm_tiled_two_epochs():
    _run_subprocess(
        """
        import torch
        import torch_nntile

        torch.manual_seed(7)
        torch_nntile.init_context(
            ncpu=1,
            ncuda=0,
            verbose=0,
            cpu_fallback=False,
        )
        torch_nntile.restrict_cpu()

        a = torch.randn(4, 8, 6)
        b = torch.randn(4, 6, 8)
        a_nnt = a.to("nntile")
        b_nnt = b.to("nntile")
        ref = torch.bmm(a, b)

        def run_round() -> torch.Tensor:
            for _ in range(3):
                c_nnt = torch.bmm(a_nnt, b_nnt)
            torch_nntile.set_axis_group_name(a_nnt, {0: "B", 1: "M", 2: "K"})
            torch_nntile.set_axis_group_name(b_nnt, {0: "B", 1: "K", 2: "N"})
            torch_nntile.set_axis_group_name(c_nnt, {0: "B", 1: "M", 2: "N"})
            torch_nntile.set_axis_group_tiling("B", 2)
            torch_nntile.set_axis_group_tiling("M", 2)
            torch_nntile.set_axis_group_tiling("N", 2)
            torch_nntile.set_axis_group_tiling("K", 2)
            torch_nntile.compile_graph()
            torch_nntile.run()
            torch_nntile.wait()
            return c_nnt

        c1 = run_round()
        c1_cpu = c1.cpu().clone()
        c2 = run_round()
        assert torch.allclose(c1_cpu, ref, rtol=1e-4, atol=1e-4)
        assert torch.allclose(c2.cpu(), ref, rtol=1e-4, atol=1e-4)
        torch_nntile.shutdown_context()
        """
    )
