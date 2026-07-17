#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_gpt2_hf.py
# Train stock HuggingFace GPT2LMHeadModel (cpu, cuda, or nntile).

"""Train HuggingFace GPT-2 on a tiny synthetic token stream.

Torch cannot use CUDA and the PrivateUse1 ``nntile`` device in one process
(PyTorch >= 2.8). Train with ``--device cpu`` / ``cuda`` / ``nntile`` in
separate runs, then ``compare`` two checkpoints. CPU is useful for small
numerical-accuracy showcases; larger runs should use cuda or nntile.

Attention: ``cuda``/``cpu`` use HF ``eager`` (classic matmul) for a fair
compare vs NNTile; ``nntile`` uses HF ``sdpa`` so attention goes through
``F.scaled_dot_product_attention``, which torch_nntile overrides with its
SDPA kernel (eager HF attention is not fully supported on nntile).

Use ``--disable-tf32`` on ``--device cuda`` for full FP32 matmul (no TF32)
when comparing numerically against nntile.

Before training, all epoch batches (inputs + labels) and the model are moved
onto the training device; the script prints prefetch time and wall training
time. On ``nntile``, each iter ``compile_graph``/``run``s after
``optimizer.zero_grad``, then prints loss via ``.to("cpu")`` (host sync) so
grad ``INVALIDATE``s share that step's compile phase (avoids ``STARPU_W``-only
clear VRAM blowup under multi-step async submit; debt D7). On ``cuda``/``cpu``,
each iter prints loss via ``.item()`` after ``zero_grad`` (device sync), and
``synchronize_device`` runs again before the final wall-time measurement.

Modes:

* ``train`` - from scratch (``--seed`` required) or resume (``--checkpoint``).
* ``compare`` - print relative Frobenius norms of weight differences.

Dataset: a deterministic synthetic token stream generated from
``--data-seed`` (defaults to ``--seed``). No external corpus is downloaded
or stored in git.

Examples::

    # From scratch on CPU (tiny accuracy showcase)
    python torch_nntile/examples/train_gpt2_hf.py train \
        --device cpu --seed 42 --config .../gpt2_hf_tiny_config.json \
        --output-dir runs/cpu --epochs 2

    # From scratch on CUDA
    python torch_nntile/examples/train_gpt2_hf.py train \
        --device cuda --seed 42 --config .../gpt2_hf_tiny_config.json \
        --output-dir runs/cuda --epochs 2

    # From scratch on nntile (separate process)
    python torch_nntile/examples/train_gpt2_hf.py train \
        --device nntile --seed 42 --config .../gpt2_hf_tiny_config.json \
        --output-dir runs/nntile --epochs 2

    # Resume from a checkpoint
    python ... train --device nntile --seed 42 \
        --checkpoint runs/nntile/checkpoint.pt \
        --output-dir runs/nntile --epochs 2

    # Compare checkpoints
    python ... compare --checkpoint-a runs/cuda/checkpoint.pt \
        --checkpoint-b runs/nntile/checkpoint.pt

Shell driver: ``run_gpt2_hf_cuda_vs_nntile.sh``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "gpt2_hf_tiny_config.json"


def _attn_implementation_for_device(device: str) -> str:
    """Attention backend for stock HF GPT-2.

    * ``cuda`` / ``cpu``: ``eager`` - classic matmul attention for a fair
      compare vs NNTile's explicit SDPA kernel (FP32 Flash / mem-efficient
      SDPA would otherwise dominate on CUDA).
    * ``nntile``: ``sdpa`` - routes through ``F.scaled_dot_product_attention``,
      which torch_nntile overrides with its SDPA kernel. Eager HF attention
      uses ops that are not fully supported on ``device=nntile``.
    """
    if device == "nntile":
        return "sdpa"
    return "eager"


def load_gpt2_config(
    path: Path,
    *,
    attn_implementation: str = "eager",
) -> GPT2Config:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    aliases = {
        "hidden_size": "n_embd",
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
        "max_position_embeddings": "n_positions",
        "intermediate_size": "n_inner",
    }
    fields = dict(raw)
    for src, dst in aliases.items():
        if src in fields and dst not in fields:
            fields[dst] = fields.pop(src)
        elif src in fields:
            fields.pop(src)
    for key in ("attn_pdrop", "resid_pdrop", "embd_pdrop"):
        fields.setdefault(key, 0.0)
    fields.setdefault("tie_word_embeddings", True)
    config = GPT2Config(**fields)
    config._attn_implementation = attn_implementation
    config.use_cache = False
    return config


def make_synthetic_tokens(
    vocab_size: int,
    *,
    num_tokens: int,
    seed: int,
) -> torch.Tensor:
    """Tiny deterministic token stream (no external dataset in git)."""
    if vocab_size < 2:
        raise ValueError("vocab_size must be >= 2")
    if num_tokens < 2:
        raise ValueError("num_tokens must be >= 2")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randint(
        0,
        vocab_size,
        (num_tokens,),
        dtype=torch.long,
        generator=generator,
    )


def build_sequences(
    token_ids: torch.Tensor,
    *,
    example_len: int,
    max_sequences: int | None,
) -> torch.Tensor:
    """Pack 1-D token ids into ``[N, example_len]`` contiguous windows.

    ``example_len`` is the raw packed length (input tokens + one next-token
    target). Model ``--seq-len`` is ``example_len - 1`` after the causal split.
    """
    if example_len < 2:
        raise ValueError("example_len must be >= 2")
    n_tokens = int(token_ids.numel())
    n_seq = n_tokens // example_len
    if n_seq == 0:
        raise ValueError(
            f"not enough tokens ({n_tokens}) for example_len={example_len}"
        )
    if max_sequences is not None:
        n_seq = min(n_seq, max_sequences)
    usable = n_seq * example_len
    return token_ids[:usable].view(n_seq, example_len)


def ensure_seq_len_fits_positions(config: GPT2Config, seq_len: int) -> None:
    """``--seq-len`` is the input/label length (positions ``0 .. seq_len-1``)."""
    if seq_len < 1:
        raise SystemExit(f"--seq-len must be >= 1, got {seq_len}")
    n_positions = int(config.n_positions)
    if seq_len > n_positions:
        raise SystemExit(
            f"--seq-len={seq_len} needs positions up to {seq_len - 1}, but "
            f"config n_positions={n_positions}. Use --seq-len <= "
            f"{n_positions}."
        )


def make_batches(
    sequences: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> list[torch.Tensor]:
    """Return a list of ``[B, T]`` int64 batches (CPU)."""
    n = sequences.shape[0]
    order = torch.arange(n)
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        order = order[torch.randperm(n, generator=generator)]
    batches: list[torch.Tensor] = []
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        if idx.numel() == 0:
            continue
        batches.append(sequences[idx].contiguous())
    return batches


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: GPT2Config, seed: int) -> GPT2LMHeadModel:
    set_seed(seed)
    config.use_cache = False
    model = GPT2LMHeadModel(config)
    model = model.float()
    model.train()
    return model


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    config: GPT2Config,
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
        "config": config.to_dict(),
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


def split_causal_batch(
    batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split packed ``[B, seq_len+1]`` into inputs/labels ``[B, seq_len]``.

    Uses ``clone()`` (not only ``contiguous()``): with ``B=1``,
    ``batch[:, 1:]`` can report ``is_contiguous()`` while keeping
    ``storage_offset != 0``, so ``contiguous()`` is a no-op.
    """
    if batch.ndim != 2 or batch.shape[1] < 2:
        raise ValueError("batch must be [B, T] with T >= 2")
    return batch[:, :-1].clone(), batch[:, 1:].clone()


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def configure_tf32(*, disable_tf32: bool, device: str) -> None:
    """Optionally disable CUDA TF32 for FP32 matmul / cuDNN.

    Useful for fair numerical compares vs NNTile FP32. No-op on cpu/nntile
    unless CUDA backends are present (then still toggles global PyTorch flags).
    """
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
    # PyTorch 2.x: 'highest' prefers full FP32 over TF32.
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    print("TF32 disabled (cuda.matmul.allow_tf32=False, cudnn.allow_tf32=False)")


def prepare_epoch_batches_cpu(
    sequences: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
    start_epoch: int,
    epochs: int,
    shuffle: bool,
) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    """Build CPU ``(inputs, labels)`` lists for every training epoch."""
    all_epochs: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
    for epoch in range(start_epoch, start_epoch + epochs):
        batches = make_batches(
            sequences,
            batch_size=batch_size,
            seed=seed + epoch,
            shuffle=shuffle,
        )
        all_epochs.append([split_causal_batch(batch) for batch in batches])
    return all_epochs


def count_batch_elems(
    epoch_batches: list[list[tuple[torch.Tensor, torch.Tensor]]],
) -> tuple[int, int]:
    """Return ``(n_input_elems, n_label_elems)`` across all epoch batches."""
    n_inputs = 0
    n_labels = 0
    for epoch_data in epoch_batches:
        for inputs, labels in epoch_data:
            n_inputs += int(inputs.numel())
            n_labels += int(labels.numel())
    return n_inputs, n_labels


@torch.no_grad()
def preload_batches_to_device(
    epoch_batches: list[list[tuple[torch.Tensor, torch.Tensor]]],
    device: torch.device,
) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    """Copy every ``(inputs, labels)`` pair onto ``device`` (cpu or cuda)."""
    out: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
    for epoch_data in epoch_batches:
        out.append(
            [
                (
                    inputs.to(device, non_blocking=True),
                    labels.to(device, non_blocking=True),
                )
                for inputs, labels in epoch_data
            ]
        )
    synchronize_device(device)
    return out


@torch.no_grad()
def preload_batches_to_nntile(
    epoch_batches: list[list[tuple[torch.Tensor, torch.Tensor]]],
) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    """Copy every ``(inputs, labels)`` pair onto ``device='nntile'``."""
    return [
        [(inputs.to("nntile"), labels.to("nntile")) for inputs, labels in epoch]
        for epoch in epoch_batches
    ]


def load_train_state(
    args: argparse.Namespace,
) -> tuple[
    GPT2Config,
    GPT2LMHeadModel,
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
    attn_impl = _attn_implementation_for_device(args.device)
    start_epoch = 0
    global_step = 0
    ckpt = None
    if args.checkpoint:
        ckpt = load_checkpoint(Path(args.checkpoint))
        config = GPT2Config.from_dict(ckpt["config"])
        config._attn_implementation = attn_impl
        config.use_cache = False
        model = GPT2LMHeadModel(config).float()
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
        config = load_gpt2_config(
            Path(args.config),
            attn_implementation=attn_impl,
        )
        model = build_model(config, args.seed)

    print(f"HF attn implementation: {attn_impl}")
    seed = int(
        args.seed
        if args.seed is not None
        else (ckpt.get("seed", 0) if ckpt else 0)
    )
    data_seed = int(args.data_seed if args.data_seed is not None else seed)
    return config, model, seed, start_epoch, global_step, data_seed, ckpt


def build_train_sequences(
    config: GPT2Config,
    args: argparse.Namespace,
    data_seed: int,
) -> torch.Tensor:
    """Pack synthetic token ids after config is final (resume-safe).

    ``--seq-len`` is the model input/label length. Each packed example is
    ``seq_len + 1`` tokens so ``split_causal_batch`` yields length ``seq_len``.
    """
    ensure_seq_len_fits_positions(config, args.seq_len)
    example_len = args.seq_len + 1
    num_tokens = example_len * (
        args.max_sequences if args.max_sequences is not None else 64
    )
    return build_sequences(
        make_synthetic_tokens(
            config.vocab_size,
            num_tokens=num_tokens,
            seed=data_seed,
        ),
        example_len=example_len,
        max_sequences=args.max_sequences,
    )


def causal_lm_loss_torch(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Next-token CE on CUDA/CPU (class dim last flattened for F.cross_entropy)."""
    logits = model(input_ids=input_ids).logits
    vocab = logits.shape[-1]
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab),
        labels.reshape(-1),
    )


def causal_lm_loss_nntile(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
):
    """Next-token CE on nntile via ``torch_nntile.training.cross_entropy``."""
    from torch_nntile.training import cross_entropy

    logits = model(input_ids=input_ids).logits
    return cross_entropy(logits, labels, reduction="mean")


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
    sequences = build_train_sequences(config, args, data_seed)
    print(
        f"Device={device.type}  sequences={sequences.shape[0]}  "
        f"seq_len={args.seq_len}  batch_size={args.batch_size}  "
        f"data_seed={data_seed}"
    )

    epoch_batches_cpu = prepare_epoch_batches_cpu(
        sequences,
        batch_size=args.batch_size,
        seed=seed,
        start_epoch=start_epoch,
        epochs=args.epochs,
        shuffle=not args.no_shuffle,
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
        "Per-iter loss .item() after zero_grad (device sync); "
        "synchronize_device before final wall time"
    )
    t_train0 = time.perf_counter()
    # Clear grads before the loop; clear again each iter before loss readout.
    optimizer.zero_grad(set_to_none=True)
    for epoch_idx, epoch_data in enumerate(epoch_batches):
        epoch = start_epoch + epoch_idx
        n_batches = len(epoch_data)
        for batch_idx, (inputs, labels) in enumerate(epoch_data):
            t_submit0 = time.perf_counter()
            loss = causal_lm_loss_torch(model, inputs, labels)
            loss.backward()
            optimizer.step()
            step_loss = loss.detach()
            del loss
            optimizer.zero_grad(set_to_none=True)
            # .item() synchronizes the device (fair vs nntile loss readout).
            loss_value = float(step_loss.item())
            del step_loss
            t_done = time.perf_counter()
            global_step += 1
            print(
                f"[{device.type}] epoch {epoch + 1}/{end_epoch}  "
                f"iter {batch_idx + 1}/{n_batches}  "
                f"loss={loss_value:.6f}  "
                f"wall={t_done - t_submit0:.3f}s  "
                f"steps={global_step}"
            )
    synchronize_device(device)
    train_wall_s = time.perf_counter() - t_train0
    print(
        f"timing torch train wall (incl. per-iter sync): {train_wall_s:.3f}s "
        f"({args.epochs} epochs)"
    )

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
    return 0


def train_nntile(args: argparse.Namespace) -> int:
    # Import only on the nntile path so CUDA/CPU training stays unaffected.
    import torch_nntile
    from torch_nntile.training import SGD, clone_model_weights

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
    sequences = build_train_sequences(config, args, data_seed)
    print(
        f"Device=nntile  sequences={sequences.shape[0]}  "
        f"seq_len={args.seq_len}  batch_size={args.batch_size}  "
        f"data_seed={data_seed}"
    )
    print(f"StarPU workers: ncpu={args.ncpu} ncuda={args.ncuda}")

    epoch_batches_cpu = prepare_epoch_batches_cpu(
        sequences,
        batch_size=args.batch_size,
        seed=seed,
        start_epoch=start_epoch,
        epochs=args.epochs,
        shuffle=not args.no_shuffle,
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

        optimizer = SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        if ckpt is not None and ckpt.get("optimizer_state_dict") is not None:
            print(
                "Note: nntile SGD velocity is not restored from checkpoint; "
                "weights were loaded."
            )

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = output_dir / "checkpoint.pt"
        end_epoch = start_epoch + args.epochs

        print("\nTraining on nntile...")
        print(
            "Per-iter compile_graph/run; loss .to('cpu') after zero_grad "
            "(sync; grad INVALIDATE in same phase)"
        )
        t_train0 = time.perf_counter()
        # Clear grads before the loop; clear again each iter before compile so
        # grad INVALIDATEs share that step's sealed phase with the train ops.
        optimizer.zero_grad(set_to_none=True)
        for epoch_idx, epoch_data in enumerate(epoch_batches):
            epoch = start_epoch + epoch_idx
            n_batches = len(epoch_data)
            for batch_idx in range(n_batches):
                inputs, labels = epoch_data[batch_idx]
                # Drop list refs so used batches can be reclaimed after run().
                epoch_data[batch_idx] = None
                t_submit0 = time.perf_counter()
                loss = causal_lm_loss_nntile(model, inputs, labels)
                loss.backward()
                optimizer.step()
                # Free autograd before compile so activation tiles unmark;
                # keep a detached scalar for host readout after zero_grad.
                # del inputs/labels is safe once their last use is recorded:
                # TensorRef drop appends ordinary graph INVALIDATE (ordered
                # after embedding); no pre-submit invalidate side channel.
                step_loss = loss.detach()
                del loss
                del inputs
                del labels
                optimizer.zero_grad(set_to_none=True)
                t_record = time.perf_counter()
                torch_nntile.compile_graph()
                t_compile = time.perf_counter()
                torch_nntile.run()
                t_run = time.perf_counter()
                # Host loss readout joins StarPU (sync point; not bare wait()).
                with torch.no_grad():
                    loss_value = float(step_loss.to("cpu").item())
                del step_loss
                t_readout = time.perf_counter()
                global_step += 1
                wall_s = t_readout - t_submit0
                line = (
                    f"[nntile] epoch {epoch + 1}/{end_epoch}  "
                    f"iter {batch_idx + 1}/{n_batches}  "
                    f"loss={loss_value:.6f}  "
                    f"wall={wall_s:.3f}s  "
                )
                if args.verbose:
                    record_s = t_record - t_submit0
                    compile_s = t_compile - t_record
                    run_s = t_run - t_compile
                    readout_s = t_readout - t_run
                    line += (
                        f"(record={record_s:.3f}s compile={compile_s:.3f}s "
                        f"run={run_s:.3f}s readout={readout_s:.3f}s)  "
                    )
                line += f"steps={global_step}"
                print(line)
        torch_nntile.wait()
        train_wall_s = time.perf_counter() - t_train0
        if args.verbose:
            torch_nntile.print_info()
        print(
            f"timing nntile train wall (incl. per-iter loss sync): "
            f"{train_wall_s:.3f}s ({args.epochs} epochs)"
        )

        weights = clone_model_weights(model)
        path = ckpt_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": weights,
            "config": config.to_dict(),
            "seed": seed,
            "epoch": end_epoch,
            "global_step": global_step,
            "device": "nntile",
            "optimizer_state_dict": None,
        }
        torch.save(payload, path)
        print(f"Saved checkpoint to {path}")
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
        help="Train GPT-2 HF on a tiny synthetic token stream",
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
            "Seed for the synthetic token stream "
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
        help="GPT-2 JSON config path",
    )
    train.add_argument("--output-dir", required=True)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--momentum", type=float, default=0.0)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help=(
            "Length of input_ids / labels after the causal split "
            "(packed examples use seq_len+1 tokens; default 32)"
        ),
    )
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument(
        "--max-sequences",
        type=int,
        default=64,
        help=(
            "Number of packed sequences in the synthetic dataset "
            "(default 64)"
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
        "--verbose",
        action="store_true",
        help=(
            "Verbose StarPU / NNTile context logging (nntile only); also "
            "print per-iter record/compile/run/readout and print_info()"
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
