# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/nntile_iter_phases.py
# Per-iteration TensorGraph: record, compile, wait, run; wait after last run.

"""Shared nntile train-phase timing.

Each step is recorded and compiled while the previous ``run()`` is still
in flight. ``wait()`` joins that previous submit, then ``run()`` starts
the compiled step. A final ``wait()`` joins the last submit. The printed
wall starts on a GPU-idle ``wait()`` **before the first record** and
runs through that final wait. Phase lines are a breakdown of the same
interval.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

LossFn = Callable[[torch.nn.Module, dict[str, torch.Tensor]], torch.Tensor]


def compile_wait_run_iter(
    torch_nntile: Any,
) -> tuple[float, float, float]:
    """Compile the pending step, wait for the previous run, then submit."""
    t0 = time.perf_counter()
    torch_nntile.compile_graph()
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    torch_nntile.wait()
    wait_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    torch_nntile.run()
    run_s = time.perf_counter() - t0
    return compile_s, wait_s, run_s


def compile_run_wait_iter(
    torch_nntile: Any,
) -> tuple[float, float, float]:
    """Compile, submit, then wait for this run (no overlap with record)."""
    t0 = time.perf_counter()
    torch_nntile.compile_graph()
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    torch_nntile.run()
    run_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    torch_nntile.wait()
    wait_s = time.perf_counter() - t0
    return compile_s, wait_s, run_s


def wait_then_start_timer(torch_nntile: Any) -> float:
    """Join NNHaul, then start the train wall before the first record."""
    torch_nntile.wait()
    return time.perf_counter()


def wait_end(torch_nntile: Any) -> float:
    """Join NNHaul after the last ``run()``."""
    t0 = time.perf_counter()
    torch_nntile.wait()
    return time.perf_counter() - t0


def print_nntile_iter_timings(
    step: int,
    n_steps: int,
    record_nntile_s: float,
    record_torch_s: float,
    compile_s: float,
    run_s: float,
    wait_s: float,
    *,
    prep_compute: bool = False,
    iter_wall_s: float | None = None,
) -> None:
    extra = ""
    if iter_wall_s is not None:
        extra = f" wall={iter_wall_s:.3f}s"
    if prep_compute:
        extra += (
            f" prep={record_nntile_s + record_torch_s + compile_s:.3f}s "
            f"compute={run_s + wait_s:.3f}s"
        )
    print(
        f"timing nntile iter {step}/{n_steps} "
        f"record(nntile)={record_nntile_s:.3f}s "
        f"record(torch)={record_torch_s:.3f}s "
        f"compile={compile_s:.3f}s run={run_s:.3f}s "
        f"wait={wait_s:.3f}s{extra}"
    )


def print_nntile_phase_timings(
    record_nntile_s: float,
    record_torch_s: float,
    compile_s: float,
    run_s: float,
    wait_s: float,
) -> None:
    print(f"timing nntile record(nntile): {record_nntile_s:.3f}s")
    print(f"timing nntile record(torch): {record_torch_s:.3f}s")
    print(f"timing nntile compile: {compile_s:.3f}s")
    print(f"timing nntile run: {run_s:.3f}s")
    print(f"timing nntile wait: {wait_s:.3f}s")


def print_nntile_prep_compute(prep_s: float, compute_s: float) -> None:
    print(f"timing nntile prep (record+compile): {prep_s:.3f}s")
    print(f"timing nntile compute (run+wait): {compute_s:.3f}s")


def measure_isolated_nntile_iter(
    torch_nntile: Any,
    record_step: Callable[[], None],
) -> None:
    """Sequential record → compile → run → wait after the train wall.

    GPU is idle (a ``wait()`` first). Phases do not overlap. Not part of
    the train wall. ``run+wait`` is submit plus join of this extra step.
    """
    torch_nntile.wait()
    nntile_t0 = torch_nntile.record_nntile_seconds()
    t0 = time.perf_counter()
    record_step()
    record_wall_s = time.perf_counter() - t0
    step_nntile_s = max(
        0.0, torch_nntile.record_nntile_seconds() - nntile_t0
    )
    step_torch_s = max(0.0, record_wall_s - step_nntile_s)
    t0 = time.perf_counter()
    torch_nntile.compile_graph()
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    torch_nntile.run()
    run_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    torch_nntile.wait()
    wait_s = time.perf_counter() - t0
    print(
        "timing nntile isolated "
        f"record(nntile)={step_nntile_s:.3f}s "
        f"record(torch)={step_torch_s:.3f}s "
        f"compile={compile_s:.3f}s run={run_s:.3f}s "
        f"wait={wait_s:.3f}s run+wait={run_s + wait_s:.3f}s"
    )
