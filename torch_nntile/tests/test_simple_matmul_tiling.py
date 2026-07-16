# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_simple_matmul_tiling.py
# Tiled matmul graph-mode regression (two compile/run epochs).

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from torch_nntile import _C
from conftest import subprocess_environ

pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)


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


def test_simple_matmul_tiled_two_epochs():
    _run_subprocess(
        """
        import torch
        import torch_nntile

        torch.manual_seed(0)
        m, n, k = 512, 384, 512
        mt, nt, kt = 256, 192, 256
        repeat = 3

        torch_nntile.init_context(
            ncpu=1,
            ncuda=0,
            verbose=0,
            cpu_fallback=False,
        )
        torch_nntile.restrict_cpu()

        a = torch.randn(m, k)
        b = torch.randn(k, n)
        a_nnt = a.to("nntile")
        b_nnt = b.to("nntile")

        def run_round() -> torch.Tensor:
            for _ in range(repeat):
                c_nnt = a_nnt @ b_nnt
            torch_nntile.set_axis_group_name(a_nnt, {0: "M", 1: "K"})
            torch_nntile.set_axis_group_name(b_nnt, {1: "N"})
            torch_nntile.set_axis_group_tiling("M", mt)
            torch_nntile.set_axis_group_tiling("N", nt)
            torch_nntile.set_axis_group_tiling("K", kt)
            torch_nntile.compile_graph()
            torch_nntile.run()
            torch_nntile.wait()
            return c_nnt

        c1 = run_round()
        c1_cpu = c1.cpu().clone()
        c2 = run_round()

        ref = a @ b
        assert torch.allclose(c1_cpu, ref, rtol=1e-4, atol=1e-4)
        assert torch.allclose(c2.cpu(), ref, rtol=1e-4, atol=1e-4)

        # Follow-up readout while session is still alive.
        _ = c2.cpu()
        torch_nntile.shutdown_context()
        """
    )
