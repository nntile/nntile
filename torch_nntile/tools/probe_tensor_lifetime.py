#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# Probe PyTorch tensor lifetime and torch_nntile recorder pinning.
# Run: python torch_nntile/tools/probe_tensor_lifetime.py [--nntile]

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import textwrap
import weakref
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class ProbeResult:
    name: str
    notes: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


def _collect() -> None:
    gc.collect()


def _tensor_alive(ref: weakref.ref) -> bool:
    return ref() is not None


def _grad_fn_chain(loss: torch.Tensor) -> list[str]:
    names: list[str] = []
    node = loss.grad_fn
    while node is not None:
        names.append(type(node).__name__)
        node = node.next_functions[0][0] if node.next_functions else None
    return names


def probe_cpu_no_grad() -> ProbeResult:
    result = ProbeResult(name="cpu_no_grad")
    with torch.no_grad():
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        c = torch.randn(4, 4)
        t_ref = None
        d = a + b + c
        # Intermediate from a+b is not bound to a Python name.
        loss = d.sum()
        result.data["loss"] = float(loss)
    _collect()
    result.notes.append(
        "no_grad: intermediates are not tracked; storage freed when refcount drops"
    )
    return result


def probe_cpu_training_chained_add() -> ProbeResult:
    result = ProbeResult(name="cpu_training_chained_add")
    saved_pack: list[str] = []
    saved_unpack: list[str] = []

    def pack(tag: str, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        for t in tensors:
            saved_pack.append(f"{tag}:{t.shape}")
        return tensors

    def unpack(tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        for t in tensors:
            saved_unpack.append(str(tuple(t.shape)))
        return tensors

    a = torch.randn(4, 4, requires_grad=True)
    b = torch.randn(4, 4, requires_grad=True)
    c = torch.randn(4, 4, requires_grad=True)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        d = a + b + c
        t = a + b
        t_ref = weakref.ref(t)
        del t
        _collect()
        result.data["intermediate_alive_after_del_named"] = _tensor_alive(t_ref)
        loss = d.sum()

    result.data["grad_fn_chain"] = _grad_fn_chain(loss)
    result.data["saved_pack_events"] = saved_pack
    result.data["saved_unpack_events"] = saved_unpack

    loss.backward()
    _collect()
    result.data["saved_unpack_after_backward"] = len(saved_unpack)
    result.notes.append(
        "add backward does not save output values; saved_tensors_hooks fire on "
        "inputs that require grad (built-in ops use SavedVariable internally)"
    )
    return result


def probe_cpu_explicit_del_intermediate() -> ProbeResult:
    result = ProbeResult(name="cpu_explicit_del_intermediate")
    a = torch.randn(4, 4, requires_grad=True)
    b = torch.randn(4, 4, requires_grad=True)
    c = torch.randn(4, 4, requires_grad=True)
    t = a + b
    t_ref = weakref.ref(t)
    d = t + c
    del t
    _collect()
    result.data["intermediate_alive_after_del"] = _tensor_alive(t_ref)
    loss = d.sum()
    loss.backward()
    _collect()
    result.data["intermediate_alive_after_backward"] = _tensor_alive(t_ref)
    result.notes.append(
        "del t drops the Python wrapper; autograd may still hold packed "
        "SavedVariable data at C++ level until backward (backward still works)"
    )
    return result


def probe_cpu_only_c_requires_grad() -> ProbeResult:
    result = ProbeResult(name="cpu_only_c_requires_grad")
    a = torch.randn(4, 4)
    b = torch.randn(4, 4)
    c = torch.randn(4, 4, requires_grad=True)
    d = a + b + c
    loss = d.sum()
    result.data["grad_fn_chain"] = _grad_fn_chain(loss)
    loss.backward()
    result.notes.append(
        "Only c requires grad: first add runs outside the grad path; "
        "intermediate (a+b) is not retained for backward"
    )
    return result


def _nntile_available() -> bool:
    try:
        import torch_nntile  # noqa: F401
        from torch_nntile import _C

        return bool(_C.has_libnntile())
    except ImportError:
        return False


def _gc_stats() -> dict[str, Any]:
    from torch_nntile import _C

    stats = _C.debug_gc_stats()
    return {
        "pinned_tensors": stats.pinned_tensors,
        "tensor_nodes": stats.tensor_nodes,
        "tile_pool": stats.tile_pool,
        "pending_ops": stats.pending_ops,
        "pending_data": stats.pending_data,
        "has_session": stats.has_session,
        "storage_releases": _C.storage_release_count(),
    }


def _run_nntile_subprocess(script: str) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    build_lib = repo / "build" / "nntile"
    starpu_lib = "/opt/starpu/lib"
    env = dict(**__import__("os").environ)
    try:
        import torch

        torch_lib = str(Path(torch.__file__).resolve().parent / "lib")
    except ImportError:
        torch_lib = ""
    ld_parts = [p for p in (str(build_lib), starpu_lib, torch_lib) if p]
    ld = env.get("LD_LIBRARY_PATH", "")
    for part in ld_parts:
        if part not in ld.split(":"):
            ld = f"{part}:{ld}" if ld else part
    env["LD_LIBRARY_PATH"] = ld
    env["STARPU_SILENT"] = "1"
    env["STARPU_FXT_TRACE"] = "0"
    env["STARPU_WORKERS_NOBIND"] = "1"
    pkg_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = f"{pkg_root}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"nntile subprocess failed ({proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return json.loads(proc.stdout.strip())


def probe_nntile_graph_mode() -> ProbeResult:
    payload = _run_nntile_subprocess(
        """
        import json
        import torch
        import torch_nntile
        from torch_nntile import _C

        def gc_stats():
            stats = _C.debug_gc_stats()
            return {
                "pinned_tensors": stats.pinned_tensors,
                "tensor_nodes": stats.tensor_nodes,
                "tile_pool": stats.tile_pool,
                "pending_ops": stats.pending_ops,
                "pending_data": stats.pending_data,
                "has_session": stats.has_session,
                "storage_releases": _C.storage_release_count(),
            }

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False, runtime_mode="graph"
        )
        torch_nntile.restrict_cpu()
        _C.reset_storage_release_count()

        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="nntile", requires_grad=True)
        b = torch.tensor([[0.5, 1.0], [1.5, 2.0]], device="nntile", requires_grad=True)
        c = torch.tensor([[0.25, 0.5], [0.75, 1.0]], device="nntile", requires_grad=True)
        out = {"after_inputs": gc_stats()}
        d = a + b + c
        out["after_forward"] = gc_stats()
        out["has_pending_graph"] = torch_nntile.has_pending_graph()
        d.backward(torch.ones_like(d))
        out["after_backward_before_execute"] = gc_stats()
        torch_nntile.compile_graph()
        out["after_compile"] = gc_stats()
        torch_nntile.run()
        out["after_run"] = gc_stats()
        del a, b, c, d
        import gc
        gc.collect()
        out["storage_releases_after_del"] = _C.storage_release_count()
        torch_nntile.shutdown_context()
        gc.collect()
        out["after_shutdown"] = gc_stats()
        print(json.dumps(out))
        """
    )
    return ProbeResult(
        name="nntile_graph_mode",
        notes=[
            "g_pinned_tensors cleared at compile; tile_pool grows until shutdown"
        ],
        data=payload,
    )


def probe_nntile_eager_mode() -> ProbeResult:
    payload = _run_nntile_subprocess(
        """
        import gc
        import json
        import torch
        import torch_nntile
        from torch_nntile import _C

        def gc_stats():
            stats = _C.debug_gc_stats()
            return {
                "pinned_tensors": stats.pinned_tensors,
                "tensor_nodes": stats.tensor_nodes,
                "tile_pool": stats.tile_pool,
                "pending_ops": stats.pending_ops,
                "pending_data": stats.pending_data,
                "has_session": stats.has_session,
                "storage_releases": _C.storage_release_count(),
            }

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False, runtime_mode="eager"
        )
        torch_nntile.restrict_cpu()
        _C.reset_storage_release_count()
        a = torch.tensor([1.0, 2.0], device="nntile")
        b = torch.tensor([3.0, 4.0], device="nntile")
        c = torch.tensor([0.5, 0.5], device="nntile")
        d = a + b + c
        out = {
            "after_forward": gc_stats(),
            "has_pending_graph": torch_nntile.has_pending_graph(),
        }
        del a, b, c, d
        gc.collect()
        out["after_del"] = gc_stats()
        out["storage_releases_after_del"] = _C.storage_release_count()
        torch_nntile.shutdown_context()
        out["after_shutdown"] = gc_stats()
        print(json.dumps(out))
        """
    )
    return ProbeResult(
        name="nntile_eager_mode",
        notes=[
            "eager mode executes and reset_recorder_locked per op batch; "
            "tile_pool cleared each execute"
        ],
        data=payload,
    )


def probe_nntile_no_grad() -> ProbeResult:
    payload = _run_nntile_subprocess(
        """
        import gc
        import json
        import torch
        import torch_nntile
        from torch_nntile import _C

        def gc_stats():
            stats = _C.debug_gc_stats()
            return {
                "pinned_tensors": stats.pinned_tensors,
                "tensor_nodes": stats.tensor_nodes,
                "tile_pool": stats.tile_pool,
                "pending_ops": stats.pending_ops,
                "pending_data": stats.pending_data,
                "has_session": stats.has_session,
                "storage_releases": _C.storage_release_count(),
            }

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False, runtime_mode="graph"
        )
        torch_nntile.restrict_cpu()
        _C.reset_storage_release_count()
        with torch.no_grad():
            a = torch.tensor([1.0, 2.0], device="nntile")
            b = torch.tensor([3.0, 4.0], device="nntile")
            c = torch.tensor([0.5, 0.5], device="nntile")
            d = a + b + c
            out = {"after_forward": gc_stats()}
        torch_nntile.compile_graph()
        torch_nntile.run()
        out["after_execute"] = gc_stats()
        gc.collect()
        out["storage_releases"] = _C.storage_release_count()
        torch_nntile.shutdown_context()
        print(json.dumps(out))
        """
    )
    return ProbeResult(name="nntile_no_grad", data=payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nntile",
        action="store_true",
        help="Include torch_nntile probes (requires built extension)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON only",
    )
    args = parser.parse_args()

    probes: list[Callable[[], ProbeResult]] = [
        probe_cpu_no_grad,
        probe_cpu_training_chained_add,
        probe_cpu_explicit_del_intermediate,
        probe_cpu_only_c_requires_grad,
    ]

    if args.nntile:
        if not _nntile_available():
            print(
                "torch_nntile with libnntile is not available; "
                "build NNTile and install torch_nntile first",
                file=sys.stderr,
            )
            return 1
        probes.extend(
            [
                probe_nntile_no_grad,
                probe_nntile_graph_mode,
                probe_nntile_eager_mode,
            ]
        )

    results = [probe() for probe in probes]
    payload = [asdict(r) for r in results]

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for item in payload:
            print(f"=== {item['name']} ===")
            for note in item.get("notes", []):
                print(f"  note: {note}")
            print(json.dumps(item.get("data", {}), indent=2))
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
