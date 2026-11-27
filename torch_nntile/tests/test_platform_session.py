# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_platform_session.py
# Platform kernel: .to() Ingress/Flush, no local compile / StarPU.

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
    flushes = []

    def ingress(nid, payload, shape, dtype):
        ingresses.append((int(nid), list(shape), str(dtype), len(payload)))

    def flush(phase_json, gather_ids, wait_only):
        phase = json.loads(phase_json)
        flushes.append(
            {
                "ops": [op["op_name"] for op in phase.get("ops") or []],
                "kinds": [
                    (op.get("attrs") or {}).get("kind")
                    for op in phase.get("ops") or []
                ],
                "gather": [int(x) for x in gather_ids],
                "wait_only": bool(wait_only),
            }
        )
        if wait_only:
            return b""
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


def test_platform_hooks_add_does_not_compile():
    stdout = _run(
        _HOOKS
        + textwrap.dedent(
            """
            a = torch.ones(2, 2).to("nntile")
            b = torch.ones(2, 2).to("nntile")
            c = (a + b).to("cpu")
            print("device", a.device.type)
            print("shape", list(c.shape))
            print("ingress", len(ingresses))
            print("flushes", json.dumps(flushes))
            print("initialized", torch_nntile.is_context_initialized())
            try:
                torch_nntile.compile_graph()
                raise SystemExit("compile_graph should raise")
            except RuntimeError as exc:
                print("compile", str(exc))
            try:
                torch_nntile.execute()
                raise SystemExit("execute should raise")
            except RuntimeError as exc:
                print("execute", str(exc))
            """
        )
    )
    assert "device nntile" in stdout or "device privateuse1" in stdout
    assert "shape [2, 2]" in stdout
    assert "ingress 2" in stdout
    assert "initialized True" in stdout
    assert "server compiles" in stdout
    payload = json.loads(stdout.split("flushes ", 1)[1].splitlines()[0])
    assert "TORCH_BINARY" in payload[0]["ops"]
    assert 4 in payload[0]["kinds"]


def test_platform_hooks_stock_linear_relu():
    stdout = _run(
        _HOOKS
        + textwrap.dedent(
            """
            torch.manual_seed(0)
            layer = torch.nn.Linear(4, 3, bias=False).float().to("nntile")
            x = torch.randn(2, 4).to("nntile")
            y = torch.relu(layer(x)).to("cpu")
            print("device", x.device.type)
            print("shape", list(y.shape))
            print("ingress", len(ingresses))
            print("flushes", json.dumps(flushes))
            """
        )
    )
    assert "shape [2, 3]" in stdout
    assert "ingress 2" in stdout
    payload = json.loads(stdout.split("flushes ", 1)[1].splitlines()[0])
    names = payload[0]["ops"]
    kinds = payload[0]["kinds"]
    assert any(n.startswith("TORCH_") for n in names)
    assert 10 in kinds
    assert 50 in kinds or 54 in kinds or 52 in kinds


def test_platform_ingress_survives_classic_gemm_no_fill():
    # Regression: the classic GEMM recorder used to emit a FILL(1.0) into
    # an ingressed operand whose node had no graph producer (platform-mode
    # ingress sends data over the wire, so no SCATTER op exists), turning
    # ``x @ w.t()`` into ``ones @ w.t()``. Ingress marks the node's data
    # as present; the flush IR must not mutate ingressed nodes.
    stdout = _run(
        _HOOKS
        + textwrap.dedent(
            """
            import torch
            from torch_nntile.nn import NntileLinear

            phases = []

            def flush_detail(phase_json, gather_ids, wait_only):
                phases.append(json.loads(phase_json))
                return b"\\x00" * 256

            _C.set_platform_hooks(ingress, flush_detail)
            torch.manual_seed(0)
            layer = NntileLinear(8, 4, bias=True).to("nntile")
            x = torch.randn(6, 8).to("nntile")
            y = layer(x)
            torch_nntile.wait()
            print("ingress", json.dumps(ingresses))
            print("phases", json.dumps(phases))
            """
        )
    )
    ingressed = json.loads(
        stdout.split("ingress ", 1)[1].splitlines()[0]
    )
    phases = json.loads(stdout.split("phases ", 1)[1].splitlines()[0])
    ingressed_ids = {entry[0] for entry in ingressed}
    x_ids = {entry[0] for entry in ingressed if entry[1] == [6, 8]}
    ops = [op for phase in phases for op in phase.get("ops") or []]
    # The GEMM must consume the ingressed x node directly.
    gemms = [op for op in ops if op.get("op_name") == "GEMM"]
    assert gemms, "no GEMM recorded for NntileLinear"
    assert any(
        op.get("inputs") and int(op["inputs"][0]) in x_ids for op in gemms
    ), f"GEMM does not read the ingressed x node: {gemms}"
    # No op may write into an ingressed node: recorder/lowering never
    # mutate wire-provided operands.
    writers = [
        op
        for op in ops
        if any(int(nid) in ingressed_ids for nid in op.get("outputs") or [])
    ]
    assert not writers, f"ops mutate ingressed nodes: {writers}"
