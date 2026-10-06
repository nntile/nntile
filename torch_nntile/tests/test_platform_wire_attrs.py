# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_platform_wire_attrs.py
# Platform wire: TORCH_BINARY GEMM-family ops must carry their packed view
# layouts on the Flush PhaseIR (stock-aten Linear backward transposes).

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

from conftest import subprocess_environ

_HOOKS = textwrap.dedent(
    r"""
    import json
    import torch
    import torch_nntile
    from torch_nntile import _C

    ingresses = []
    phases = []

    def ingress(nid, payload, shape, dtype):
        ingresses.append((int(nid), list(shape), str(dtype)))

    def flush(phase_json, gather_ids, wait_only):
        phases.append(json.loads(phase_json))
        return b"\x00" * 256

    _C.set_platform_hooks(ingress, flush)
    """
)


def _run(script: str) -> str:
    env = subprocess_environ()
    env.pop("NNTILE_SERVER_SOCKET", None)
    env["NNTILE_ENABLE_LOCAL_COMPILER"] = ""
    proc = subprocess.run(
        [sys.executable, "-c", script],
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
    return proc.stdout


def _torch_ops(phases: list[dict]) -> list[dict]:
    return [
        op
        for phase in phases
        for op in phase.get("ops") or []
        if str(op.get("op_name", "")).startswith("TORCH")
    ]


def test_mm_backward_flush_keeps_transposed_layout():
    # Regression: the PhaseIR used to encode only attrs.kind for
    # TORCH_BINARY ops, dropping the packed view layout of
    # mm(grad.t(), x) recorded by stock-aten Linear backward. The
    # daemon then read grad as a contiguous [out, batch] matrix and
    # failed with "mat1 and mat2 shapes cannot be multiplied" (or
    # worse, computed a wrong product where shapes still type-checked)
    # - nntile/platform zoo README gap 2.
    stdout = _run(
        _HOOKS
        + textwrap.dedent(
            """
            torch.manual_seed(0)
            model = torch.nn.Sequential(
                torch.nn.Linear(8, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 4),
            ).float().to("nntile")
            x = torch.randn(6, 8).to("nntile")
            y = model(x)
            y.backward(torch.ones_like(y))
            torch_nntile.wait()
            print("phases", json.dumps(phases))
            """
        )
    )
    phases = json.loads(stdout.split("phases ", 1)[1].splitlines()[0])
    ops = _torch_ops(phases)
    mm_ops = [
        op
        for op in ops
        if op.get("op_name") == "TORCH_BINARY"
        and (op.get("attrs") or {}).get("kind") == 50
    ]
    assert mm_ops, f"no TORCH_BINARY Mm in flushes: {ops}"
    transposed = 0
    for op in mm_ops:
        layouts = (op.get("attrs") or {}).get("layouts") or []
        for layout in layouts:
            if layout.get("arg") != "in":
                continue
            sizes = layout.get("sizes") or []
            strides = layout.get("strides") or []
            if len(sizes) == 2 and len(strides) == 2 and sizes[1] != 0:
                # Row-major contiguous: strides == [sizes[1], 1]. A
                # column-major read of the storage node is the
                # transposed view the mm needs.
                if strides[0] == 1 and strides[1] != 1:
                    transposed += 1
    assert transposed >= 1, (
        "backward mm flush carries no transposed view layout; "
        f"mm ops: {mm_ops}"
    )


def test_addmm_forward_layout_roundtrip_shapes():
    # The forward Linear records Addmm (kind 52); its packed layouts
    # must survive too so the daemon reads the same operands.
    stdout = _run(
        _HOOKS
        + textwrap.dedent(
            """
            torch.manual_seed(0)
            model = torch.nn.Linear(8, 4).float().to("nntile")
            x = torch.randn(6, 8).to("nntile")
            y = model(x)
            torch_nntile.wait()
            print("phases", json.dumps(phases))
            """
        )
    )
    phases = json.loads(stdout.split("phases ", 1)[1].splitlines()[0])
    ops = _torch_ops(phases)
    addmm = [
        op
        for op in ops
        if (op.get("op_name") == "TORCH_TERNARY"
            and (op.get("attrs") or {}).get("kind") == 52)
        or (op.get("op_name") == "TORCH_BINARY"
            and (op.get("attrs") or {}).get("kind") in (50, 54))
    ]
    assert addmm, f"no Linear-family op in flushes: {ops}"
    for op in addmm:
        layouts = (op.get("attrs") or {}).get("layouts") or []
        assert layouts, f"Linear-family op lacks layouts: {op}"
        for layout in layouts:
            assert layout.get("arg") in ("in", "out")
            assert len(layout.get("sizes") or []) == len(
                layout.get("strides") or []
            )
