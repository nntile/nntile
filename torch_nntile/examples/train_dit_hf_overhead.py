#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_dit_hf_overhead.py
# Train stock HuggingFace Diffusers DiTTransformer2DModel (cpu, cuda, or nntile).

"""Train HuggingFace Diffusers DiT on a tiny synthetic diffusion batch stream.

Torch cannot use CUDA and the PrivateUse1 ``nntile`` device in one process
(PyTorch >= 2.8). Train with ``--device cpu`` / ``cuda`` / ``nntile`` in
separate runs, then ``compare`` two checkpoints.

Diffusion: stock Diffusers ``DiTTransformer2DModel`` noise-prediction MSE (``diffusion_mse_loss``). ``--disable-tf32`` only disables TF32 GEMM / cuDNN matmul on CUDA.

Before training, all epoch batches (inputs + labels) and the model are moved
onto the training device; the script prints prefetch time and wall training
time. On ``nntile`` the prefetch ``wait()``s so StarPU H2D finishes before
optimizer setup. Both paths then ``wait`` / ``synchronize_device`` again
**immediately before the train timer**, so leftover compute is not in
flight when the clock starts. Loss uses stock MSE noise-prediction (same on CUDA and nntile). On nntile,
each iter is **recorded** and ``compile_graph``'d while the previous
``run()`` is in flight, then ``wait()`` joins that submit and ``run()``
starts the compiled step. A final ``wait()`` joins the last submit.
``--wait-after-run`` instead does ``run()`` then ``wait()`` on the same
step so record/compile never overlap GPU work (prep vs compute).
Cumulative record / compile / run / wait times are printed as a
breakdown. The train wall is that clock through the final ``wait()``
(every record, compile, wait, and run). The final loss is read after
that join. CUDA synchronizes after every iter so each printed step
includes device work, then reads the final loss after the wall.
After the loss (and checkpoint), one extra isolated step is timed:
CUDA prints a synchronized iter wall; nntile prints sequential
record / compile / run / wait / run+wait with the GPU idle (no
overlap with a previous ``run()``).

Examples::

    python torch_nntile/examples/train_dit_hf_overhead.py train \\
        --device cuda --disable-tf32 --seed 42 \\
        --config torch_nntile/examples/overhead_dit/dit_xs.json \\
        --batch-size 1 --max-sequences 10 --epochs 1 \\
        --output-dir /tmp/dit_overhead_s_cuda

Overhead ladder: ``torch_nntile/examples/overhead_dit/``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from hf_tiny_train_common import configure_single_thread_host
from nntile_iter_phases import (
    compile_run_wait_iter,
    compile_wait_run_iter,
    print_nntile_prep_compute,
    measure_isolated_nntile_iter,
    print_nntile_iter_timings,
    print_nntile_phase_timings,
    print_torch_isolated_iter,
    print_torch_iter_timings,
    wait_end,
    wait_then_start_timer,
)
from dit_hf_tiny_train_common import (
    config_to_dict,
    disable_dit_label_dropout,
    diffusion_mse_loss,
    make_synthetic_diffusion_batch,
)
from diffusers import DiTTransformer2DModel
from typing import Any


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "overhead_dit/dit_xs.json"


BatchDict = dict[str, torch.Tensor]


def build_train_batches(
    config: Any,
    args: argparse.Namespace,
    data_seed: int,
) -> list[BatchDict]:
    n_steps = args.max_sequences if args.max_sequences is not None else 64
    sample_size = int(config.sample_size)
    in_channels = int(config.in_channels)
    num_timesteps = int(config.num_embeds_ada_norm)
    num_classes = max(num_timesteps, 10)
    return [
        make_synthetic_diffusion_batch(
            batch_size=args.batch_size,
            sample_size=sample_size,
            in_channels=in_channels,
            num_timesteps=num_timesteps,
            num_classes=num_classes,
            seed=data_seed + step,
        )
        for step in range(n_steps)
    ]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(fields: dict[str, Any], seed: int) -> DiTTransformer2DModel:
    set_seed(seed)
    model = DiTTransformer2DModel(**fields).float().train()
    disable_dit_label_dropout(model)
    return model


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    config: Any,
    seed: int,
    epoch: int,
    global_step: int,
    optimizer_state: dict | None,
    device_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
    payload = {
        "model_state_dict": state,
        "config": config_to_dict(config),
        "seed": seed,
        "epoch": epoch,
        "global_step": global_step,
        "device": device_name,
        "optimizer_state_dict": optimizer_state,
    }
    torch.save(payload, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"invalid checkpoint format: {path}")
    return payload


def relative_frobenius(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> float:
    """``||a - b||_F / max(||a||_F, ||b||_F, eps)``."""
    with torch.no_grad():
        diff = (a.float() - b.float()).norm().item()
        na = a.float().norm().item()
        nb = b.float().norm().item()
    return diff / max(na, nb, eps)


def compare_checkpoints(path_a: Path, path_b: Path) -> int:
    ckpt_a = load_checkpoint(path_a)
    ckpt_b = load_checkpoint(path_b)
    state_a = ckpt_a["model_state_dict"]
    state_b = ckpt_b["model_state_dict"]
    keys_a = set(state_a)
    keys_b = set(state_b)
    if keys_a != keys_b:
        only_a = sorted(keys_a - keys_b)
        only_b = sorted(keys_b - keys_a)
        print("WARNING: state_dict key mismatch")
        if only_a:
            print(f"  only in A ({len(only_a)}): {only_a[:8]}")
        if only_b:
            print(f"  only in B ({len(only_b)}): {only_b[:8]}")
    shared = sorted(keys_a & keys_b)
    print(f"Comparing {len(shared)} tensors")
    print(f"  A: {path_a}")
    print(f"  B: {path_b}")
    max_rel = 0.0
    worst = ""
    for name in shared:
        ta = state_a[name]
        tb = state_b[name]
        if ta.shape != tb.shape:
            print(
                f"  SKIP {name}: shape {tuple(ta.shape)} vs "
                f"{tuple(tb.shape)}"
            )
            continue
        rel = relative_frobenius(ta, tb)
        if rel > max_rel:
            max_rel = rel
            worst = name
        print(f"  {name}: relative_frobenius={rel:.6e}")
    print(f"max relative_frobenius={max_rel:.6e}  ({worst})")
    return 0


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def configure_tf32(*, disable_tf32: bool, device: str) -> None:
    if not disable_tf32:
        return
    if device != "cuda":
        print(
            f"Note: --disable-tf32 is mainly for --device cuda "
            f"(got --device {device}); applying PyTorch CUDA TF32 flags anyway "
            "if CUDA backends exist."
        )
    if hasattr(torch.backends, "cuda") and hasattr(
        torch.backends.cuda, "matmul"
    ):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    print("TF32 disabled (cuda.matmul.allow_tf32=False, cudnn.allow_tf32=False)")


def prepare_epoch_batches_cpu(
    batches: list[BatchDict],
    *,
    epochs: int,
) -> list[list[BatchDict]]:
    return [list(batches) for _ in range(epochs)]


def count_batch_elems(
    epoch_batches: list[list[BatchDict]],
) -> tuple[int, int]:
    n_inputs = 0
    n_labels = 0
    for epoch_data in epoch_batches:
        for batch in epoch_data:
            n_inputs += int(batch["noisy"].numel())
            n_labels += int(batch["noise"].numel())
    return n_inputs, n_labels


@torch.no_grad()
def preload_batches_to_device(
    epoch_batches: list[list[BatchDict]],
    device: torch.device,
) -> list[list[BatchDict]]:
    out: list[list[BatchDict]] = []
    for epoch_data in epoch_batches:
        out.append(
            [
                {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                for batch in epoch_data
            ]
        )
    synchronize_device(device)
    return out


@torch.no_grad()
def preload_batches_to_nntile(
    epoch_batches: list[list[BatchDict]],
) -> list[list[BatchDict]]:
    return [
        [
            {k: v.to("nntile") for k, v in batch.items()}
            for batch in epoch_data
        ]
        for epoch_data in epoch_batches
    ]


def load_train_state(
    args: argparse.Namespace,
) -> tuple[
    Any,
    DiTTransformer2DModel,
    int,
    int,
    int,
    int,
    dict | None,
]:
    """Load config/model/sequences metadata for a train run.

    Returns
    ``(config, cpu_model, seed, start_epoch, global_step, data_seed, ckpt)``.
    """
    start_epoch = 0
    global_step = 0
    ckpt = None
    if args.checkpoint:
        ckpt = load_checkpoint(Path(args.checkpoint))
        model = DiTTransformer2DModel.from_config(ckpt["config"]).float()
        model.load_state_dict(ckpt["model_state_dict"])
        disable_dit_label_dropout(model)
        model.train()
        config = model.config
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        print(
            f"Resumed from {args.checkpoint} "
            f"(epoch={start_epoch}, step={global_step})"
        )
    else:
        if args.seed is None:
            raise SystemExit("--seed is required when training from scratch")
        fields = json.load(Path(args.config).open(encoding="utf-8"))
        fields = {
            k: v for k, v in fields.items() if not str(k).startswith("_")
        }
        model = build_model(fields, args.seed)
        config = model.config
    seed = int(
        args.seed
        if args.seed is not None
        else (ckpt.get("seed", 0) if ckpt else 0)
    )
    data_seed = int(args.data_seed if args.data_seed is not None else seed)
    return config, model, seed, start_epoch, global_step, data_seed, ckpt


def dit_loss(model: DiTTransformer2DModel, batch: BatchDict) -> torch.Tensor:
    return diffusion_mse_loss(model, batch)


def _nntile_only_args_set(args: argparse.Namespace) -> list[str]:
    """Return nntile-only CLI flags that were explicitly set."""
    ignored: list[str] = []
    if args.ncpu != -1:
        ignored.append(f"--ncpu={args.ncpu}")
    if args.ncuda != -1:
        ignored.append(f"--ncuda={args.ncuda}")
    if args.restrict_cuda:
        ignored.append("--restrict-cuda")
    if args.restrict_cpu:
        ignored.append("--restrict-cpu")
    if args.verbose:
        ignored.append("--verbose")
    return ignored


def train_torch(args: argparse.Namespace) -> int:
    """Pure-PyTorch training on ``--device cpu`` or ``cuda``."""
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. Use a CUDA build of PyTorch and a GPU, "
            "or train with --device cpu / nntile."
        )

    ignored = _nntile_only_args_set(args)
    if ignored:
        print(
            "Ignoring nntile-only arguments on "
            f"--device {args.device}: {', '.join(ignored)}"
        )

    (
        config,
        model,
        seed,
        start_epoch,
        global_step,
        data_seed,
        ckpt,
    ) = load_train_state(args)
    batches = build_train_batches(config, args, data_seed)
    print(
        f"Device={device.type}  sequences={len(batches)}  "
        f"sample_size={int(config.sample_size)}  patches={(int(config.sample_size) // int(config.patch_size)) ** 2}  "
        f"batch_size={args.batch_size}  "
        f"data_seed={data_seed}"
    )

    epoch_batches_cpu = prepare_epoch_batches_cpu(
        batches,
        epochs=args.epochs,
    )
    n_input_elems, n_label_elems = count_batch_elems(epoch_batches_cpu)

    print(f"Prefetching batches + model to {device}...")
    t_pre0 = time.perf_counter()
    with torch.no_grad():
        epoch_batches = preload_batches_to_device(epoch_batches_cpu, device)
        model = model.to(device)
        synchronize_device(device)
    prefetch_s = time.perf_counter() - t_pre0
    print(
        f"timing host->{device} prefetch: {prefetch_s:.3f}s "
        f"(input elems {n_input_elems}, label elems {n_label_elems}, + model)"
    )
    del epoch_batches_cpu

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    if ckpt is not None:
        opt_state = ckpt.get("optimizer_state_dict")
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint.pt"
    end_epoch = start_epoch + args.epochs

    print(f"\nTraining on torch ({device})...")
    print(
        "Per-iter synchronize so each printed step includes GPU work; "
        "loss readout after the wall"
    )
    optimizer.zero_grad(set_to_none=True)
    last_loss: torch.Tensor | None = None
    last_batch: BatchDict | None = None
    n_epoch_batches = len(epoch_batches)
    n_steps = sum(len(epoch_data) for epoch_data in epoch_batches)
    synchronize_device(device)
    t_train0 = time.perf_counter()
    print(
        "timing torch train wall t0: device idle, "
        "clock includes first iter through last synchronize",
        flush=True,
    )
    for epoch_idx, epoch_data in enumerate(epoch_batches):
        n_batches = len(epoch_data)
        for batch_idx, batch in enumerate(epoch_data):
            t_iter0 = time.perf_counter()
            loss = dit_loss(model, batch)
            loss.backward()
            optimizer.step()
            step_loss = loss.detach()
            del loss
            optimizer.zero_grad(set_to_none=True)
            synchronize_device(device)
            iter_s = time.perf_counter() - t_iter0
            global_step += 1
            is_last = (
                epoch_idx == n_epoch_batches - 1
                and batch_idx == n_batches - 1
            )
            if is_last:
                last_loss = step_loss
                last_batch = batch
            else:
                del step_loss
            print_torch_iter_timings(global_step, n_steps, iter_s)
    train_wall_s = time.perf_counter() - t_train0
    if last_loss is None:
        raise RuntimeError("train_torch: no steps ran")
    loss_value = float(last_loss.item())
    del last_loss
    print(
        f"[{device.type}] final loss={loss_value:.6f}  steps={global_step}"
    )
    print(
        f"timing torch train wall (loop+sync, loss readout after): "
        f"{train_wall_s:.3f}s ({args.epochs} epochs)"
    )

    if not args.no_checkpoint:
        save_checkpoint(
            ckpt_path,
            model=model,
            config=config,
            seed=seed,
            epoch=end_epoch,
            global_step=global_step,
            optimizer_state=optimizer.state_dict(),
            device_name=device.type,
        )
    if last_batch is None:
        raise RuntimeError("train_torch: missing last batch")
    print(
        "Isolated extra step after loss (not in train wall; "
        "GPU idle, synchronized)"
    )
    synchronize_device(device)
    t_iso0 = time.perf_counter()
    loss = dit_loss(model, last_batch)
    loss.backward()
    optimizer.step()
    del loss
    optimizer.zero_grad(set_to_none=True)
    synchronize_device(device)
    print_torch_isolated_iter(time.perf_counter() - t_iso0)
    return 0


def train_nntile(args: argparse.Namespace) -> int:
    # Import only on the nntile path so CUDA/CPU training stays unaffected.
    import torch_nntile

    if args.restrict_cuda and args.restrict_cpu:
        raise SystemExit("Pass only one of --restrict-cuda / --restrict-cpu")

    (
        config,
        cpu_model,
        seed,
        start_epoch,
        global_step,
        data_seed,
        ckpt,
    ) = load_train_state(args)
    batches = build_train_batches(config, args, data_seed)
    print(
        f"Device=nntile  sequences={len(batches)}  "
        f"sample_size={int(config.sample_size)}  patches={(int(config.sample_size) // int(config.patch_size)) ** 2}  "
        f"batch_size={args.batch_size}  "
        f"data_seed={data_seed}"
    )
    print(f"StarPU workers: ncpu={args.ncpu} ncuda={args.ncuda}")

    epoch_batches_cpu = prepare_epoch_batches_cpu(
        batches,
        epochs=args.epochs,
    )
    n_input_elems, n_label_elems = count_batch_elems(epoch_batches_cpu)

    torch_nntile.init_context(
        ncpu=args.ncpu,
        ncuda=args.ncuda,
        verbose=int(args.verbose),
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
        # Submit H2D and join before the train timer so the loop only
        # records compute on already-resident nntile tensors.
        torch_nntile.compile_graph()
        torch_nntile.run()
        torch_nntile.wait()
        prefetch_s = time.perf_counter() - t_pre0
        print(
            f"timing host->nntile prefetch: {prefetch_s:.3f}s "
            f"(input elems {n_input_elems}, label elems {n_label_elems}, "
            f"+ model)"
        )
        del cpu_model
        del epoch_batches_cpu
        for param in model.parameters():
            param.requires_grad_(True)

        # Stock SGD records aten add_/mul_ into TensorGraph (torch-native).
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        if ckpt is not None and ckpt.get("optimizer_state_dict") is not None:
            opt_state = ckpt.get("optimizer_state_dict")
            if opt_state is not None:
                try:
                    optimizer.load_state_dict(opt_state)
                except (ValueError, RuntimeError) as exc:
                    print(
                        "Note: could not restore optimizer state "
                        f"({exc}); weights were loaded."
                    )

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = output_dir / "checkpoint.pt"
        end_epoch = start_epoch + args.epochs

        print("\nTraining on nntile...")
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
        if torch_nntile.has_pending_graph():
            torch_nntile.compile_graph()
            torch_nntile.run()
        last_loss: torch.Tensor | None = None
        last_batch: BatchDict | None = None
        n_epoch_batches = len(epoch_batches)
        n_steps = sum(len(epoch_data) for epoch_data in epoch_batches)
        record_nntile_s = 0.0
        record_torch_s = 0.0
        compile_s = 0.0
        run_s = 0.0
        wait_s = 0.0
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
                loss = dit_loss(model, batch)
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
        if args.verbose:
            torch_nntile.print_info()
        if last_loss is None:
            raise RuntimeError("train_nntile: no steps ran")
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

        if not args.no_checkpoint:
            with torch.no_grad():
                weights = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
            path = ckpt_path
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model_state_dict": weights,
                "config": config_to_dict(config),
                "seed": seed,
                "epoch": end_epoch,
                "global_step": global_step,
                "device": "nntile",
                "optimizer_state_dict": None,
            }
            torch.save(payload, path)
            print(f"Saved checkpoint to {path}")
        if last_batch is None:
            raise RuntimeError("train_nntile: missing last batch")
        print(
            "Isolated extra step after loss (not in train wall; "
            "GPU idle, sequential record/compile/run/wait)"
        )

        def _record_isolated() -> None:
            loss = dit_loss(model, last_batch)
            loss.backward()
            optimizer.step()
            del loss
            optimizer.zero_grad(set_to_none=True)

        measure_isolated_nntile_iter(torch_nntile, _record_isolated)
    finally:
        torch_nntile.shutdown_context()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser(
        "train",
        help="Train Diffusers DiT HF on a tiny synthetic diffusion stream",
    )
    train.add_argument(
        "--device",
        required=True,
        choices=("cpu", "cuda", "nntile"),
        help=(
            "Training device (cpu/cuda/nntile need separate processes; "
            "cpu is for small numerical-accuracy showcases)"
        ),
    )
    train.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (required when training from scratch)",
    )
    train.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help=(
            "Seed for the synthetic diffusion stream "
            "(default: same as --seed / checkpoint seed)"
        ),
    )
    train.add_argument(
        "--checkpoint",
        default="",
        help="Resume weights from this checkpoint",
    )
    train.add_argument(
        "--config",
        default=str(_default_config_path()),
        help="DiT JSON config path",
    )
    train.add_argument("--output-dir", required=True)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--momentum", type=float, default=0.0)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument(
        "--max-sequences",
        type=int,
        default=64,
        help=(
            "Number of diffusion training steps (default 64)"
        ),
    )
    train.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable per-epoch shuffle",
    )
    train.add_argument(
        "--disable-tf32",
        action="store_true",
        help=(
            "Disable CUDA TF32 for matmul/cuDNN (full FP32). "
            "Recommended for fair numerical compares vs nntile FP32 "
            "(applies on --device cuda)"
        ),
    )
    train.add_argument(
        "--ncpu",
        type=int,
        default=-1,
        help="StarPU CPU workers for nntile (ignored on --device cpu/cuda)",
    )
    train.add_argument(
        "--ncuda",
        type=int,
        default=-1,
        help="StarPU CUDA workers for nntile (ignored on --device cpu/cuda)",
    )
    train.add_argument(
        "--restrict-cuda",
        action="store_true",
        help=(
            "Pin nntile kernels to CUDA workers "
            "(ignored on --device cpu/cuda)"
        ),
    )
    train.add_argument(
        "--restrict-cpu",
        action="store_true",
        help=(
            "Pin nntile kernels to CPU workers "
            "(ignored on --device cpu/cuda)"
        ),
    )
    train.add_argument(
        "--wait-after-run",
        action="store_true",
        help=(
            "Nntile only: wait() immediately after each run() so record "
            "and compile do not overlap GPU work. Prints prep vs compute."
        ),
    )
    train.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Skip writing checkpoint.pt (overhead benchmarks)",
    )
    train.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Verbose StarPU / NNTile context logging (nntile only); also "
            "print_info() after run()+wait()"
        ),
    )

    compare = sub.add_parser(
        "compare",
        help="Print relative Frobenius norms between two checkpoints",
    )
    compare.add_argument("--checkpoint-a", required=True)
    compare.add_argument("--checkpoint-b", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compare":
        return compare_checkpoints(
            Path(args.checkpoint_a),
            Path(args.checkpoint_b),
        )
    if args.command == "train":
        if not args.checkpoint and args.seed is None:
            raise SystemExit("--seed is required when training from scratch")
        configure_single_thread_host()
        configure_tf32(
            disable_tf32=bool(args.disable_tf32),
            device=args.device,
        )
        if args.device == "nntile":
            return train_nntile(args)
        return train_torch(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
