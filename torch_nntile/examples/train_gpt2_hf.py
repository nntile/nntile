#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_gpt2_hf.py
# Train stock HuggingFace GPT2LMHeadModel (cuda or nntile).

"""Train HuggingFace GPT-2 on a tiny synthetic token stream.

Torch cannot use CUDA and the PrivateUse1 ``nntile`` device in one process
(PyTorch >= 2.8). Train with ``--device cuda`` or ``--device nntile`` in
separate runs, then ``compare`` two checkpoints.

Modes:

* ``train`` — from scratch (``--seed`` required) or resume (``--checkpoint``).
* ``compare`` — print relative Frobenius norms of weight differences.

Dataset: a deterministic synthetic token stream generated from
``--data-seed`` (defaults to ``--seed``). No external corpus is downloaded
or stored in git.

Examples::

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
import sys
from pathlib import Path

import torch
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
    fields.setdefault("tie_word_embeddings", True)
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
    seq_len: int,
    max_sequences: int | None,
) -> torch.Tensor:
    """Pack 1-D token ids into ``[N, seq_len]`` contiguous windows."""
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    n_tokens = int(token_ids.numel())
    n_seq = n_tokens // seq_len
    if n_seq == 0:
        raise ValueError(
            f"not enough tokens ({n_tokens}) for seq_len={seq_len}"
        )
    if max_sequences is not None:
        n_seq = min(n_seq, max_sequences)
    usable = n_seq * seq_len
    return token_ids[:usable].view(n_seq, seq_len)


def ensure_seq_len_fits_positions(config: GPT2Config, seq_len: int) -> None:
    """Causal LM uses positions ``0 .. seq_len-2`` after the next-token split."""
    n_positions = int(config.n_positions)
    # After split_causal_batch, input length is seq_len - 1.
    if seq_len - 1 > n_positions:
        raise SystemExit(
            f"--seq-len={seq_len} needs positions up to {seq_len - 2}, but "
            f"config n_positions={n_positions}. Use --seq-len <= "
            f"{n_positions + 1}."
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
    """Split ``[B, T]`` into inputs ``[:, :-1]`` and labels ``[:, 1:]`` on CPU.

    Uses ``clone()`` (not only ``contiguous()``): with ``B=1``,
    ``batch[:, 1:]`` can report ``is_contiguous()`` while keeping
    ``storage_offset != 0``, so ``contiguous()`` is a no-op.
    """
    if batch.ndim != 2 or batch.shape[1] < 2:
        raise ValueError("batch must be [B, T] with T >= 2")
    return batch[:, :-1].clone(), batch[:, 1:].clone()


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


def train_cuda(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. Use a CUDA build of PyTorch and a GPU, "
            "or train with --device nntile."
        )
    device = torch.device("cuda")
    start_epoch = 0
    global_step = 0
    ckpt = None
    if args.checkpoint:
        ckpt = load_checkpoint(Path(args.checkpoint))
        config = GPT2Config.from_dict(ckpt["config"])
        config._attn_implementation = "sdpa"
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
        config = load_gpt2_config(Path(args.config))
        model = build_model(config, args.seed)

    seed = int(
        args.seed
        if args.seed is not None
        else (ckpt.get("seed", 0) if ckpt else 0)
    )
    # Pack sequences after config is final (checkpoint vocab_size on resume).
    ensure_seq_len_fits_positions(config, args.seq_len)
    data_seed = int(
        args.data_seed if args.data_seed is not None else seed
    )
    num_tokens = args.seq_len * (
        args.max_sequences if args.max_sequences is not None else 64
    )
    sequences = build_sequences(
        make_synthetic_tokens(
            config.vocab_size,
            num_tokens=num_tokens,
            seed=data_seed,
        ),
        seq_len=args.seq_len,
        max_sequences=args.max_sequences,
    )
    print(
        f"Device=cuda  sequences={sequences.shape[0]}  "
        f"seq_len={args.seq_len}  batch_size={args.batch_size}  "
        f"data_seed={data_seed}"
    )

    model = model.to(device)
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

    for epoch in range(start_epoch, start_epoch + args.epochs):
        batches = make_batches(
            sequences,
            batch_size=args.batch_size,
            seed=seed + epoch,
            shuffle=not args.no_shuffle,
        )
        running = 0.0
        for batch in batches:
            inputs_cpu, labels_cpu = split_causal_batch(batch)
            inputs = inputs_cpu.to(device)
            labels = labels_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = causal_lm_loss_torch(model, inputs, labels)
            loss.backward()
            optimizer.step()
            global_step += 1
            running += float(loss.detach().item())
        mean_loss = running / max(len(batches), 1)
        print(
            f"[cuda] epoch {epoch + 1}/{start_epoch + args.epochs}  "
            f"loss={mean_loss:.6f}  steps={global_step}"
        )

    save_checkpoint(
        ckpt_path,
        model=model,
        config=config,
        seed=seed,
        epoch=start_epoch + args.epochs,
        global_step=global_step,
        optimizer_state=optimizer.state_dict(),
        device_name="cuda",
    )
    return 0


def train_nntile(args: argparse.Namespace) -> int:
    # Import only on the nntile path so CUDA training stays unaffected.
    root = _repo_root() / "torch_nntile"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import torch_nntile
    from torch_nntile import _C
    from torch_nntile.training import SGD, clone_model_weights

    if not _C.has_libnntile():
        raise SystemExit(
            "torch_nntile was built without libnntile. "
            "Set NNTILE_BUILD_DIR and reinstall."
        )

    start_epoch = 0
    global_step = 0
    ckpt = None
    if args.checkpoint:
        ckpt = load_checkpoint(Path(args.checkpoint))
        config = GPT2Config.from_dict(ckpt["config"])
        config._attn_implementation = "sdpa"
        config.use_cache = False
        cpu_model = GPT2LMHeadModel(config).float()
        cpu_model.load_state_dict(ckpt["model_state_dict"])
        cpu_model.train()
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
        cpu_model = build_model(config, args.seed)

    seed = int(
        args.seed
        if args.seed is not None
        else (ckpt.get("seed", 0) if ckpt else 0)
    )
    # Pack sequences after config is final (checkpoint vocab_size on resume).
    ensure_seq_len_fits_positions(config, args.seq_len)
    data_seed = int(
        args.data_seed if args.data_seed is not None else seed
    )
    num_tokens = args.seq_len * (
        args.max_sequences if args.max_sequences is not None else 64
    )
    sequences = build_sequences(
        make_synthetic_tokens(
            config.vocab_size,
            num_tokens=num_tokens,
            seed=data_seed,
        ),
        seq_len=args.seq_len,
        max_sequences=args.max_sequences,
    )
    print(
        f"Device=nntile  sequences={sequences.shape[0]}  "
        f"seq_len={args.seq_len}  batch_size={args.batch_size}  "
        f"data_seed={data_seed}"
    )

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
        with torch.no_grad():
            model = cpu_model.to("nntile")
        del cpu_model
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

        for epoch in range(start_epoch, start_epoch + args.epochs):
            batches = make_batches(
                sequences,
                batch_size=args.batch_size,
                seed=seed + epoch,
                shuffle=not args.no_shuffle,
            )
            running = 0.0
            for batch_cpu in batches:
                inputs_cpu, labels_cpu = split_causal_batch(batch_cpu)
                with torch.no_grad():
                    inputs = inputs_cpu.to("nntile")
                    labels = labels_cpu.to("nntile")
                optimizer.zero_grad(set_to_none=True)
                loss = causal_lm_loss_nntile(model, inputs, labels)
                loss.backward()
                optimizer.step()
                torch_nntile.compile_graph()
                torch_nntile.run()
                torch_nntile.wait()
                with torch.no_grad():
                    loss_val = float(loss.to("cpu").item())
                running += loss_val
                global_step += 1
            mean_loss = running / max(len(batches), 1)
            print(
                f"[nntile] epoch {epoch + 1}/{start_epoch + args.epochs}  "
                f"loss={mean_loss:.6f}  steps={global_step}"
            )

        weights = clone_model_weights(model)
        path = ckpt_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": weights,
            "config": config.to_dict(),
            "seed": seed,
            "epoch": start_epoch + args.epochs,
            "global_step": global_step,
            "device": "nntile",
            "optimizer_state_dict": None,
        }
        torch.save(payload, path)
        print(f"Saved checkpoint to {path}")
    finally:
        torch_nntile.wait()
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
        choices=("cuda", "nntile"),
        help="Training device (cuda and nntile need separate processes)",
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
    train.add_argument("--seq-len", type=int, default=32)
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
    train.add_argument("--verbose", action="store_true")

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
        if args.device == "cuda":
            return train_cuda(args)
        return train_nntile(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
