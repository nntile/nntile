#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_gpt2.py
# Train NNTile-optimized GPT2LMHead on device="nntile" (HF-format checkpoints).

"""Train the NNTile-layout minimal GPT-2 on a tiny synthetic token stream.

Always uses ``device="nntile"`` (requires ``torch_nntile`` ops such as
``SDPA`` / ``gemm``). Init and checkpoints use HuggingFace GPT-2 weight
layout so losses and weights can be compared against
``train_gpt2_hf.py --device cuda``.

Torch cannot use CUDA and the PrivateUse1 ``nntile`` device in one process
(PyTorch >= 2.8). Train this script and ``train_gpt2_hf.py`` separately,
then ``compare`` the two HF-format checkpoints.

Before training, all epoch batches (inputs + labels) and the model are moved
onto ``nntile``; the script prints prefetch time and wall training time.
Each iter ``compile_graph``/``run``s after ``optimizer.zero_grad``, then
prints loss via ``.to("cpu")`` (host sync) so grad ``INVALIDATE``s share that
step's compile phase. A bare multi-step async ``run()`` without sync would
let ``STARPU_W``-only clears allocate one working set per in-flight step
(see debt D7 in ``docs/dev/torch_nntile_tensor_architecture.md``).

No axis tiling yet - full tensors on nntile.

Modes:

* ``train`` - from scratch (``--seed`` required) or resume (``--checkpoint``).
* ``compare`` - print relative Frobenius norms of weight differences.

Dataset: a deterministic synthetic token stream generated from
``--data-seed`` (defaults to ``--seed``). No external corpus is downloaded
or stored in git.

Examples::

    # From scratch on nntile
    python torch_nntile/examples/train_gpt2.py train \\
        --seed 42 --config .../gpt2_hf_tiny_config.json \\
        --output-dir runs/nntile_minimal --epochs 2

    # Resume from an HF-format checkpoint
    python ... train --seed 42 \\
        --checkpoint runs/nntile_minimal/checkpoint.pt \\
        --output-dir runs/nntile_minimal --epochs 2

    # Compare against HF CUDA training
    python ... compare --checkpoint-a runs/cuda/checkpoint.pt \\
        --checkpoint-b runs/nntile_minimal/checkpoint.pt

Shell driver: ``run_gpt2_cuda_vs_nntile.sh``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch_nntile
from transformers import GPT2Config, GPT2LMHeadModel


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "gpt2_hf_tiny_config.json"


def load_gpt2_config(path: Path) -> GPT2Config:
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
    fields.setdefault("tie_word_embeddings", False)
    config = GPT2Config(**fields)
    config._attn_implementation = "sdpa"
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
def preload_batches_to_nntile(
    epoch_batches: list[list[tuple[torch.Tensor, torch.Tensor]]],
) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    """Copy every ``(inputs, labels)`` pair onto ``device='nntile'``."""
    return [
        [(inputs.to("nntile"), labels.to("nntile")) for inputs, labels in epoch]
        for epoch in epoch_batches
    ]


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


def build_minimal_from_hf(
    config: GPT2Config,
    hf: GPT2LMHeadModel,
):
    """Construct ``GPT2LMHead`` and copy HF weights (CPU)."""
    from torch_nntile.models.gpt2_hf_loader import load_hf_into_gpt2_lm_head
    from torch_nntile.models.gpt2_minimal import GPT2LMHead

    model = GPT2LMHead(config).float()
    load_hf_into_gpt2_lm_head(model, hf)
    model.train()
    return model


def load_train_state(
    args: argparse.Namespace,
):
    """Load config / CPU minimal model / metadata for a train run.

    Init matches ``train_gpt2_hf.py``: HF ``GPT2LMHeadModel`` is created with
    ``--seed`` (or loaded from an HF-format checkpoint), then converted into
    NNTile-layout ``GPT2LMHead``.

    Returns
    ``(config, cpu_model, seed, start_epoch, global_step, data_seed, ckpt)``.
    """
    start_epoch = 0
    global_step = 0
    ckpt = None
    if args.checkpoint:
        ckpt = load_checkpoint(Path(args.checkpoint))
        config = GPT2Config.from_dict(ckpt["config"])
        config._attn_implementation = "sdpa"
        config.use_cache = False
        hf = GPT2LMHeadModel(config).float()
        missing, unexpected = hf.load_state_dict(
            ckpt["model_state_dict"],
            strict=False,
        )
        if unexpected:
            print(f"WARNING: unexpected checkpoint keys: {unexpected[:8]}")
        if missing:
            print(f"WARNING: missing checkpoint keys: {missing[:8]}")
        hf.train()
        model = build_minimal_from_hf(config, hf)
        del hf
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        print(
            f"Resumed from {args.checkpoint} "
            f"(epoch={start_epoch}, step={global_step})"
        )
    else:
        if args.seed is None:
            raise SystemExit("--seed is required when training from scratch")
        config = load_gpt2_config(Path(args.config))
        set_seed(args.seed)
        config.use_cache = False
        hf = GPT2LMHeadModel(config).float()
        hf.train()
        model = build_minimal_from_hf(config, hf)
        del hf

    seed = int(
        args.seed
        if args.seed is not None
        else (ckpt.get("seed", 0) if ckpt else 0)
    )
    data_seed = int(args.data_seed if args.data_seed is not None else seed)
    return config, model, seed, start_epoch, global_step, data_seed, ckpt


def causal_lm_loss_nntile(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
):
    """Next-token CE on nntile via ``torch_nntile.training.cross_entropy``."""
    from torch_nntile.training import cross_entropy

    logits = model(input_ids)
    return cross_entropy(logits, labels, reduction="mean")


def save_hf_checkpoint(
    path: Path,
    *,
    model,
    config: GPT2Config,
    seed: int,
    epoch: int,
    global_step: int,
) -> None:
    """Save NNTile model weights as an HF-format ``checkpoint.pt``."""
    from torch_nntile.models.gpt2_hf_loader import (
        export_gpt2_lm_head_to_hf_state_dict,
    )
    from torch_nntile.models.gpt2_minimal import GPT2LMHead
    from torch_nntile.training import clone_model_weights

    weights = clone_model_weights(model)
    cpu_model = GPT2LMHead(config).float()
    cpu_model.load_state_dict(weights)
    hf_state = export_gpt2_lm_head_to_hf_state_dict(cpu_model, config=config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": hf_state,
        "config": config.to_dict(),
        "seed": seed,
        "epoch": epoch,
        "global_step": global_step,
        "device": "nntile",
        "optimizer_state_dict": None,
        "model_impl": "gpt2_minimal",
    }
    torch.save(payload, path)
    print(f"Saved HF-format checkpoint to {path}")


def train_nntile(args: argparse.Namespace) -> int:
    from torch_nntile.training import SGD

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
        f"Device=nntile (GPT2LMHead)  sequences={sequences.shape[0]}  "
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
        # Seal ingress scatters now so the first train compile is O(step),
        # not O(model+batches). Do not wait - overlap with setup.
        torch_nntile.compile_graph()
        torch_nntile.run()
        del cpu_model
        del epoch_batches_cpu
        for param in model.parameters():
            param.requires_grad_(True)

        batch_sizes = sorted(
            {
                int(inputs.size(0))
                for epoch_data in epoch_batches
                for inputs, _labels in epoch_data
            }
        )
        model.warm_sequence_caches(
            batch_sizes=batch_sizes,
            seq_len=int(args.seq_len),
            device="nntile",
        )
        print(
            "Cached position_ids / causal_mask on nntile for "
            f"batch_sizes={batch_sizes}, seq_len={args.seq_len}"
        )

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

        print("\nTraining on nntile (GPT2LMHead)...")
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

        save_hf_checkpoint(
            ckpt_path,
            model=model,
            config=config,
            seed=seed,
            epoch=end_epoch,
            global_step=global_step,
        )
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
        help="Train NNTile GPT-2 on a tiny synthetic token stream",
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
        help="Resume weights from this HF-format checkpoint",
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
        "--ncpu",
        type=int,
        default=-1,
        help="StarPU CPU workers",
    )
    train.add_argument(
        "--ncuda",
        type=int,
        default=-1,
        help="StarPU CUDA workers",
    )
    train.add_argument(
        "--restrict-cuda",
        action="store_true",
        help="Pin nntile kernels to CUDA workers",
    )
    train.add_argument(
        "--restrict-cpu",
        action="store_true",
        help="Pin nntile kernels to CPU workers",
    )
    train.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Verbose StarPU / NNTile context logging; also print per-iter "
            "record/compile/run/readout and print_info() after training"
        ),
    )

    compare = sub.add_parser(
        "compare",
        help="Print relative Frobenius norms between two HF-format checkpoints",
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
        return train_nntile(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
