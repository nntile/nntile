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
    """Join StarPU, then start the train wall before the first record."""
    torch_nntile.wait()
    return time.perf_counter()


def wait_end(torch_nntile: Any) -> float:
    """Join StarPU after the last ``run()``."""
    t0 = time.perf_counter()
    torch_nntile.wait()
    return time.perf_counter() - t0


def print_torch_iter_timings(
    step: int,
    n_steps: int,
    wall_s: float,
) -> None:
    print(
        f"timing torch iter {step}/{n_steps} wall={wall_s:.3f}s"
    )


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
) -> None:
    extra = ""
    if prep_compute:
        extra = (
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


def print_torch_isolated_iter(wall_s: float) -> None:
    print(f"timing torch isolated iter wall={wall_s:.3f}s")


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


def run_nntile_train_iters(
    *,
    name: str,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    loss_fn: LossFn,
    steps: int,
    opt: torch.optim.Optimizer,
    torch_nntile: Any,
) -> int:
    """Record/compile each step, wait, then run; wait after the last run."""
    if torch_nntile.has_pending_graph():
        torch_nntile.compile_graph()
        torch_nntile.run()
    record_nntile_s = 0.0
    record_torch_s = 0.0
    compile_s = 0.0
    run_s = 0.0
    wait_s = 0.0
    last_loss: torch.Tensor | None = None
    t_train0 = wait_then_start_timer(torch_nntile)
    for step in range(steps):
        nntile_t0 = torch_nntile.record_nntile_seconds()
        t0 = time.perf_counter()
        loss = loss_fn(model, batch)
        loss.backward()
        opt.step()
        step_loss = loss.detach()
        del loss
        opt.zero_grad(set_to_none=True)
        record_wall_s = time.perf_counter() - t0
        step_nntile_s = max(
            0.0, torch_nntile.record_nntile_seconds() - nntile_t0
        )
        record_nntile_s += step_nntile_s
        record_torch_s += max(0.0, record_wall_s - step_nntile_s)
        dc, dw, dr = compile_wait_run_iter(torch_nntile)
        compile_s += dc
        wait_s += dw
        run_s += dr
        if step == steps - 1:
            extra_wait = wait_end(torch_nntile)
            wait_s += extra_wait
            dw += extra_wait
            last_loss = step_loss
        else:
            del step_loss
        print_nntile_iter_timings(
            step + 1,
            steps,
            step_nntile_s,
            max(0.0, record_wall_s - step_nntile_s),
            dc,
            dr,
            dw,
        )
    wall_s = time.perf_counter() - t_train0
    if last_loss is None:
        raise RuntimeError(f"{name}: no steps ran")
    with torch.no_grad():
        loss_val = float(last_loss.to("cpu").item())
    del last_loss
    print_nntile_phase_timings(
        record_nntile_s, record_torch_s, compile_s, run_s, wait_s
    )
    torch_nntile.print_info()
    print(f"[{name}] final loss={loss_val:.6f}")
    print(f"[{name}] wall={wall_s:.3f}s  OK")
    return 0
