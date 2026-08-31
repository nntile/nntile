# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/nntile_native_overhead_common.py
# Shared 10-step overhead loop for torch_nntile.models (classic kernels).

"""Train wall matching ``train_gpt2.py``: dict batches, classic SGD / CE."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

import torch_nntile
from nntile_iter_phases import (
    compile_run_wait_iter,
    compile_wait_run_iter,
    measure_isolated_nntile_iter,
    print_nntile_iter_timings,
    print_nntile_phase_timings,
    print_nntile_prep_compute,
    wait_end,
    wait_then_start_timer,
)
from torch_nntile.training import SGD

BatchDict = dict[str, torch.Tensor]
LossFn = Callable[[torch.nn.Module, BatchDict], torch.Tensor]


def count_batch_elems(
    epoch_batches: list[list[BatchDict]],
) -> tuple[int, int]:
    n_inputs = 0
    n_labels = 0
    for epoch_data in epoch_batches:
        for batch in epoch_data:
            n_inputs += int(batch["input_ids"].numel())
            n_labels += int(batch["labels"].numel())
    return n_inputs, n_labels


@torch.no_grad()
def preload_batches_to_nntile(
    epoch_batches: list[list[BatchDict]],
) -> list[list[BatchDict]]:
    out: list[list[BatchDict]] = []
    for epoch in epoch_batches:
        loaded: list[BatchDict] = []
        for batch in epoch:
            loaded.append({k: v.to("nntile") for k, v in batch.items()})
        out.append(loaded)
    return out


def _warm_sequence_caches(
    model: torch.nn.Module,
    *,
    batch_sizes: list[int],
    seq_len: int,
) -> None:
    target: Any = model
    if not hasattr(target, "warm_sequence_caches"):
        for attr in ("transformer", "model", "gpt_neox"):
            inner = getattr(model, attr, None)
            if inner is not None and hasattr(inner, "warm_sequence_caches"):
                target = inner
                break
    if not hasattr(target, "warm_sequence_caches"):
        return
    target.warm_sequence_caches(
        batch_sizes=batch_sizes,
        seq_len=seq_len,
        device="nntile",
    )
    print(
        "Cached sequence tables on nntile for "
        f"batch_sizes={batch_sizes}, seq_len={seq_len}"
    )


def run_native_overhead(
    *,
    name: str,
    args: Any,
    cpu_model: torch.nn.Module,
    epoch_batches_cpu: list[list[BatchDict]],
    loss_fn: LossFn,
    seq_len: int,
) -> int:
    if args.restrict_cuda and args.restrict_cpu:
        raise SystemExit("Pass only one of --restrict-cuda / --restrict-cpu")

    n_input_elems, n_label_elems = count_batch_elems(epoch_batches_cpu)
    print(
        f"Device=nntile ({name})  steps="
        f"{sum(len(e) for e in epoch_batches_cpu)}  "
        f"seq_len={seq_len}  batch_size={args.batch_size}"
    )
    print(f"StarPU workers: ncpu={args.ncpu} ncuda={args.ncuda}")

    torch_nntile.init_context(
        ncpu=args.ncpu,
        ncuda=args.ncuda,
        verbose=int(getattr(args, "verbose", 0)),
        cpu_fallback=False,
    )
    if args.restrict_cuda:
        torch_nntile.restrict_cuda()
        print("Worker placement: CUDA only (restrict_cuda)")
    elif args.restrict_cpu:
        torch_nntile.restrict_cpu()
        print("Worker placement: CPU only (restrict_cpu)")

    try:
        print("Prefetching batches + model to nntile...")
        t_pre0 = time.perf_counter()
        with torch.no_grad():
            epoch_batches = preload_batches_to_nntile(epoch_batches_cpu)
            model = cpu_model.to("nntile")
        prefetch_s = time.perf_counter() - t_pre0
        print(
            f"timing host->nntile prefetch: {prefetch_s:.3f}s "
            f"(input elems {n_input_elems}, label elems {n_label_elems}, "
            f"+ model)"
        )
        torch_nntile.compile_graph()
        torch_nntile.run()
        del cpu_model
        del epoch_batches_cpu
        for param in model.parameters():
            param.requires_grad_(True)

        batch_sizes = sorted(
            {
                int(batch["input_ids"].size(0))
                for epoch_data in epoch_batches
                for batch in epoch_data
            }
        )
        _warm_sequence_caches(
            model, batch_sizes=batch_sizes, seq_len=seq_len
        )
        if torch_nntile.has_pending_graph():
            torch_nntile.compile_graph()
            torch_nntile.run()
        torch_nntile.wait()

        optimizer = SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            momentum=getattr(args, "momentum", 0.0),
            weight_decay=getattr(args, "weight_decay", 0.0),
        )
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        print(f"\nTraining on nntile ({name})...")
        if args.wait_after_run:
            print(
                "Per-iter record, compile_graph, run, wait "
                "(wait joins this run; no overlap with record/compile)"
            )
        else:
            print(
                "Per-iter record, compile_graph, wait, run "
                "(wait joins the previous run)"
            )
        optimizer.zero_grad(set_to_none=True)
        last_loss: torch.Tensor | None = None
        last_batch: BatchDict | None = None
        n_epoch_batches = len(epoch_batches)
        n_steps = sum(len(epoch_data) for epoch_data in epoch_batches)
        record_nntile_s = 0.0
        record_torch_s = 0.0
        compile_s = 0.0
        run_s = 0.0
        wait_s = 0.0
        global_step = 0
        t_train0 = wait_then_start_timer(torch_nntile)
        print(
            "timing nntile train wall t0: GPU idle, "
            "clock includes first record through final wait",
            flush=True,
        )
        first_record_logged = False
        for epoch_idx, epoch_data in enumerate(epoch_batches):
            n_batches = len(epoch_data)
            for batch_idx in range(n_batches):
                batch = epoch_data[batch_idx]
                epoch_data[batch_idx] = None
                nntile_t0 = torch_nntile.record_nntile_seconds()
                t_record0 = time.perf_counter()
                loss = loss_fn(model, batch)
                loss.backward()
                optimizer.step()
                step_loss = loss.detach()
                del loss
                optimizer.zero_grad(set_to_none=True)
                record_wall_s = time.perf_counter() - t_record0
                if not first_record_logged:
                    print(
                        "timing nntile elapsed after first record: "
                        f"{time.perf_counter() - t_train0:.3f}s "
                        "(must be > 0 if the wall includes that record)",
                        flush=True,
                    )
                    first_record_logged = True
                step_nntile_s = max(
                    0.0,
                    torch_nntile.record_nntile_seconds() - nntile_t0,
                )
                step_torch_s = max(0.0, record_wall_s - step_nntile_s)
                record_nntile_s += step_nntile_s
                record_torch_s += step_torch_s
                if args.wait_after_run:
                    dc, dw, dr = compile_run_wait_iter(torch_nntile)
                else:
                    dc, dw, dr = compile_wait_run_iter(torch_nntile)
                compile_s += dc
                wait_s += dw
                run_s += dr
                global_step += 1
                is_last = (
                    epoch_idx == n_epoch_batches - 1
                    and batch_idx == n_batches - 1
                )
                if is_last:
                    if not args.wait_after_run:
                        extra_wait = wait_end(torch_nntile)
                        wait_s += extra_wait
                        dw += extra_wait
                    last_loss = step_loss
                    last_batch = batch
                else:
                    del step_loss
                    del batch
                print_nntile_iter_timings(
                    global_step,
                    n_steps,
                    step_nntile_s,
                    step_torch_s,
                    dc,
                    dr,
                    dw,
                    prep_compute=args.wait_after_run,
                )
        train_wall_s = time.perf_counter() - t_train0
        if last_loss is None:
            raise RuntimeError("native overhead: no steps ran")
        with torch.no_grad():
            loss_value = float(last_loss.to("cpu").item())
        del last_loss
        print_nntile_phase_timings(
            record_nntile_s, record_torch_s, compile_s, run_s, wait_s
        )
        if args.wait_after_run:
            print_nntile_prep_compute(
                record_nntile_s + record_torch_s + compile_s,
                run_s + wait_s,
            )
        print(f"[nntile] final loss={loss_value:.6f}  steps={global_step}")
        print(
            f"timing nntile train wall "
            f"(loop through final wait, loss readout after): "
            f"{train_wall_s:.3f}s ({args.epochs} epochs)"
        )
        if last_batch is None:
            raise RuntimeError("native overhead: missing last batch")
        print(
            "Isolated extra step after loss (not in train wall; "
            "GPU idle, sequential record/compile/run/wait)"
        )

        def _record_isolated() -> None:
            loss = loss_fn(model, last_batch)
            loss.backward()
            optimizer.step()
            del loss
            optimizer.zero_grad(set_to_none=True)

        measure_isolated_nntile_iter(torch_nntile, _record_isolated)
    finally:
        torch_nntile.shutdown_context()
    return 0
