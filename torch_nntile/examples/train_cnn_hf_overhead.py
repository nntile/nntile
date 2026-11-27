#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_cnn_hf_overhead.py
# Train stock torch.nn CNNs on cpu / cuda / nntile for overhead ladders.

"""Train stock torch.nn CNN graphs on a synthetic image stream.

There is no ``torch_nntile.nn.model`` CNN rewrite. This script only measures
**HF(cuda)** vs **HF(nntile)**: the same ``nn.Module`` on ``device=cuda``
versus ``device=nntile``. ``HF`` here means the stock PyTorch CNN
implementation (LeNet / ResNet / VGG / MobileNet / U-Net / modern U-Net),
not HuggingFace Transformers.

Torch cannot use CUDA and the PrivateUse1 ``nntile`` device in one process
(PyTorch >= 2.8). Train with ``--device cpu`` / ``cuda`` / ``nntile`` in
separate runs.

On ``nntile`` each iter is recorded and ``compile_graph``'d while the
previous ``run()`` is in flight, then ``wait()`` joins that submit and
``run()`` starts the compiled step. A final ``wait()`` joins the last
submit. ``--wait-after-run`` does ``run()`` then ``wait()`` on the same
step (prep vs compute). CUDA synchronizes after every iter. Prefetch is
outside the train wall. After the loss, one extra isolated step is timed.

Examples::

    python torch_nntile/examples/train_cnn_hf_overhead.py train \\
        --model resnet --device cuda --disable-tf32 --disable-cudnn \\
        --seed 42 \\
        --config torch_nntile/examples/overhead_resnet/resnet_xs.json \\
        --batch-size 1 --max-sequences 10 --epochs 1 --no-checkpoint \\
        --output-dir /tmp/resnet_overhead_xs_cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from cnn_tiny_train_common import (
    classification_ce_loss, make_image_batch, make_segmentation_batch,
    segmentation_ce_loss)
from hf_tiny_train_common import (
    compare_checkpoints, configure_cudnn, configure_single_thread_host,
    configure_tf32, load_checkpoint, load_json_object)
from nntile_iter_phases import (
    compile_run_wait_iter, compile_wait_run_iter, measure_isolated_nntile_iter,
    print_nntile_iter_timings, print_nntile_phase_timings,
    print_nntile_prep_compute, print_torch_isolated_iter,
    print_torch_iter_timings, wait_end, wait_then_start_timer)

BatchDict = dict[str, torch.Tensor]
ModelFactory = Callable[[dict[str, Any]], nn.Module]
LossFn = Callable[[nn.Module, BatchDict], torch.Tensor]


CNN_MODELS = (
    "lenet",
    "resnet",
    "vgg",
    "mobilenet",
    "unet",
    "unet_modern",
)


def _model_factory(model_name: str) -> tuple[ModelFactory, bool]:
    """Return ``(ctor, is_segmentation)`` for a stock CNN family."""
    if model_name == "lenet":
        from train_lenet_tiny import TinyLeNet

        return TinyLeNet, False
    if model_name == "resnet":
        from train_resnet_tiny import TinyResNet

        return TinyResNet, False
    if model_name == "vgg":
        from train_vgg_tiny import TinyVGG

        return TinyVGG, False
    if model_name == "mobilenet":
        from train_mobilenet_tiny import TinyMobileNet

        return TinyMobileNet, False
    if model_name == "unet":
        from train_unet_tiny import TinyUNet

        return TinyUNet, True
    if model_name == "unet_modern":
        from train_unet_modern_tiny import TinyModernUNet

        return TinyModernUNet, True
    raise SystemExit(f"unknown --model {model_name}")


def _default_config_path(model_name: str) -> Path:
    return (
        Path(__file__).resolve().parent
        / f"overhead_{model_name}"
        / f"{model_name}_xs.json"
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    model_cls: ModelFactory,
    cfg: dict[str, Any],
    seed: int,
) -> nn.Module:
    set_seed(seed)
    return model_cls(cfg).float().train()


def cnn_loss(
    model: nn.Module,
    batch: BatchDict,
    *,
    segmentation: bool,
) -> torch.Tensor:
    if segmentation:
        return segmentation_ce_loss(model, batch)
    return classification_ce_loss(model, batch)


def build_train_batches(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    data_seed: int,
    *,
    segmentation: bool,
) -> list[BatchDict]:
    n_steps = args.max_sequences if args.max_sequences is not None else 64
    maker = make_segmentation_batch if segmentation else make_image_batch
    return [
        maker(
            batch_size=args.batch_size,
            channels=int(cfg["in_channels"]),
            height=int(cfg["height"]),
            width=int(cfg["width"]),
            num_classes=int(cfg["num_classes"]),
            seed=data_seed + step,
        )
        for step in range(n_steps)
    ]


def count_params(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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
            n_inputs += int(batch["images"].numel())
            n_labels += int(batch["labels"].numel())
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


def reset_cuda_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def log_cuda_allocator_peak() -> None:
    """Allocator peak only. Board peak is sampled by a sidecar process."""
    alloc = torch.cuda.max_memory_allocated() / 1024.0**3
    print(f"cuda_max_allocated_gib={alloc:.2f}", flush=True)


def load_train_state(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], nn.Module, int, int, int, int, dict | None]:
    model_cls, _seg = _model_factory(args.model)
    start_epoch = 0
    global_step = 0
    ckpt = None
    if args.checkpoint:
        ckpt = load_checkpoint(Path(args.checkpoint))
        cfg = dict(ckpt["config"])
        model = model_cls(cfg).float()
        model.load_state_dict(ckpt["model_state_dict"])
        model.train()
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        print(
            f"Resumed from {args.checkpoint} "
            f"(epoch={start_epoch}, step={global_step})"
        )
    else:
        if args.seed is None:
            raise SystemExit("--seed is required when training from scratch")
        cfg = load_json_object(Path(args.config))
        model = build_model(model_cls, cfg, args.seed)
    seed = int(
        args.seed
        if args.seed is not None
        else (ckpt.get("seed", 0) if ckpt else 0)
    )
    data_seed = int(args.data_seed if args.data_seed is not None else seed)
    return cfg, model, seed, start_epoch, global_step, data_seed, ckpt


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    config: dict[str, Any],
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
        "config": dict(config),
        "seed": seed,
        "epoch": epoch,
        "global_step": global_step,
        "device": device_name,
        "optimizer_state_dict": optimizer_state,
    }
    torch.save(payload, path)
    print(f"Saved checkpoint to {path}")


def _nntile_only_args_set(args: argparse.Namespace) -> list[str]:
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


def _describe_cfg(cfg: dict[str, Any]) -> str:
    h = int(cfg["height"])
    w = int(cfg["width"])
    return f"spatial={h}x{w}"


def sgd_lr(args: argparse.Namespace, cfg: dict[str, Any]) -> float:
    """JSON ``lr`` overrides ``--lr`` so wide ResNet configs can stay stable."""
    if "lr" in cfg:
        return float(cfg["lr"])
    return float(args.lr)


def train_torch(args: argparse.Namespace) -> int:
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
    _, segmentation = _model_factory(args.model)
    (
        cfg,
        model,
        seed,
        start_epoch,
        global_step,
        data_seed,
        ckpt,
    ) = load_train_state(args)
    batches = build_train_batches(
        cfg, args, data_seed, segmentation=segmentation
    )
    n_params = count_params(model)
    lr = sgd_lr(args, cfg)
    print(
        f"Device={device.type}  model={args.model}  "
        f"sequences={len(batches)}  {_describe_cfg(cfg)}  "
        f"batch_size={args.batch_size}  "
        f"params={n_params} ({n_params * 4 / 1024**3:.2f} GiB FP32)  "
        f"lr={lr:g}  "
        f"data_seed={data_seed}"
    )
    epoch_batches_cpu = prepare_epoch_batches_cpu(
        batches, epochs=args.epochs
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
        lr=lr,
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
    if device.type == "cuda":
        reset_cuda_peak_vram()
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
            loss = cnn_loss(model, batch, segmentation=segmentation)
            loss.backward()
            optimizer.step()
            step_loss = loss.detach()
            del loss
            synchronize_device(device)
            optimizer.zero_grad(set_to_none=True)
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
    if device.type == "cuda":
        log_cuda_allocator_peak()
    if not args.no_checkpoint:
        save_checkpoint(
            ckpt_path,
            model=model,
            config=cfg,
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
    loss = cnn_loss(model, last_batch, segmentation=segmentation)
    loss.backward()
    optimizer.step()
    del loss
    optimizer.zero_grad(set_to_none=True)
    synchronize_device(device)
    print_torch_isolated_iter(time.perf_counter() - t_iso0)
    return 0


def train_nntile(args: argparse.Namespace) -> int:
    import torch_nntile

    if args.restrict_cuda and args.restrict_cpu:
        raise SystemExit("Pass only one of --restrict-cuda / --restrict-cpu")
    _, segmentation = _model_factory(args.model)
    (
        cfg,
        cpu_model,
        seed,
        start_epoch,
        global_step,
        data_seed,
        ckpt,
    ) = load_train_state(args)
    batches = build_train_batches(
        cfg, args, data_seed, segmentation=segmentation
    )
    n_params = count_params(cpu_model)
    lr = sgd_lr(args, cfg)
    print(
        f"Device=nntile  model={args.model}  "
        f"sequences={len(batches)}  {_describe_cfg(cfg)}  "
        f"batch_size={args.batch_size}  "
        f"params={n_params} ({n_params * 4 / 1024**3:.2f} GiB FP32)  "
        f"lr={lr:g}  "
        f"data_seed={data_seed}"
    )
    print(f"StarPU workers: ncpu={args.ncpu} ncuda={args.ncuda}")
    epoch_batches_cpu = prepare_epoch_batches_cpu(
        batches, epochs=args.epochs
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
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
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
                loss = cnn_loss(model, batch, segmentation=segmentation)
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
                "config": dict(cfg),
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
            loss_i = cnn_loss(
                model, last_batch, segmentation=segmentation
            )
            loss_i.backward()
            optimizer.step()
            del loss_i
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
        help="Train a stock torch.nn CNN on a synthetic image stream",
    )
    train.add_argument(
        "--model",
        required=True,
        choices=CNN_MODELS,
        help="CNN family (stock torch.nn graph, not torch_nntile.nn.model)",
    )
    train.add_argument(
        "--device",
        required=True,
        choices=("cpu", "cuda", "nntile"),
        help=(
            "Training device (cpu/cuda/nntile need separate processes)"
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
        help="Seed for the synthetic stream (default: same as --seed)",
    )
    train.add_argument("--checkpoint", default="")
    train.add_argument(
        "--config",
        default="",
        help="JSON model config path (default: overhead_<model>/<model>_xs.json)",
    )
    train.add_argument("--output-dir", required=True)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help=(
            "SGD learning rate. A JSON config key ``lr`` overrides this "
            "(ResNet overhead configs use 1e-4 so B=1 BN does not explode)."
        ),
    )
    train.add_argument("--momentum", type=float, default=0.0)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument(
        "--max-sequences",
        type=int,
        default=10,
        help="Number of optimizer steps (default 10)",
    )
    train.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Kept for parity with HF overhead CLIs (batches are pre-built)",
    )
    train.add_argument(
        "--disable-tf32",
        action="store_true",
        help=(
            "Disable CUDA TF32 (cuBLAS matmul, cuDNN conv/RNN). "
            "Without this flag, TF32 is enabled on those paths. "
            "Honored by ATen CUDA kernels on --device nntile too."
        ),
    )
    train.add_argument(
        "--disable-cudnn",
        action="store_true",
        help=(
            "Set torch.backends.cudnn.enabled=False so HF(cuda) "
            "BatchNorm / conv use the same ATen kernels as nntile "
            "(not cuDNN). Apply on both --device cuda and nntile."
        ),
    )
    train.add_argument("--ncpu", type=int, default=-1)
    train.add_argument("--ncuda", type=int, default=-1)
    train.add_argument("--restrict-cuda", action="store_true")
    train.add_argument("--restrict-cpu", action="store_true")
    train.add_argument(
        "--wait-after-run",
        action="store_true",
        help=(
            "Nntile only: wait() immediately after each run() so record "
            "and compile do not overlap GPU work."
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
        help="Verbose StarPU / NNTile context logging (nntile only)",
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
        if not args.config:
            args.config = str(_default_config_path(args.model))
        configure_single_thread_host()
        configure_tf32(
            disable_tf32=bool(args.disable_tf32),
            device=args.device,
        )
        configure_cudnn(disable_cudnn=bool(args.disable_cudnn))
        if args.device == "nntile":
            return train_nntile(args)
        return train_torch(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
