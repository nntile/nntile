#!/usr/bin/env python3
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file python/examples/gpt2_training.py
# Port of nntile/examples/gpt2_graph_training.cc
#
# @version 1.1.0

"""GPT-2 causal LM training on the graph API."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import nntile
from nntile import (
    AdamW,
    CausalLmBatch,
    CausalLmBatchConfig,
    CausalLmBatchIterator,
    Context,
    DataType,
    Gpt2Causal,
    NNGraph,
    TokenMemoryMap,
    init_random_parameter_hints,
    make_tiny_gpt2_config,
    sync_param_hint_from_runtime,
)


def fill_arange_position_ids(
    pos_data: np.ndarray, n_seq: int, n_batch: int,
) -> None:
    for b in range(n_batch):
        for s in range(n_seq):
            pos_data[s + n_seq * b] = s


def sdpa_causal_mask_bool_fortran_fill(n_seq: int) -> np.ndarray:
    """BOOL causal mask, Fortran layout: out[kk + n_seq * qq] = (kk <= qq)."""
    out = np.zeros(n_seq * n_seq, dtype=np.uint8)
    for qq in range(n_seq):
        for kk in range(n_seq):
            if kk <= qq:
                out[kk + n_seq * qq] = 1
    return out


def scheduled_lr(step: int, args: argparse.Namespace) -> float:
    if args.warmup_steps <= 0:
        return args.lr
    if step < args.warmup_steps:
        return args.lr * float(step + 1) / float(args.warmup_steps)
    return args.lr


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='GPT-2 graph training')
    p.add_argument('--train-bin', required=True, help='uint16 train.bin path')
    p.add_argument('--config', default='', help='GPT-2 JSON config path')
    p.add_argument('--load-weights', default='', help='SafeTensors weights')
    p.add_argument('--output-dir', default='', help='checkpoint directory')
    p.add_argument('--seq', type=int, default=8)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tiny', action='store_true')
    p.add_argument('--shuffle', action='store_true')
    p.add_argument(
        '--max-batches',
        type=int,
        default=1,
        help='Max training batches per epoch (default 1 for Python v1)',
    )
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--beta2', type=float, default=0.999)
    p.add_argument('--warmup-steps', type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.epochs < 1:
        print('gpt2_training: --epochs must be >= 1', file=sys.stderr)
        return 1

    n_seq = args.seq
    n_batch = args.batch

    if args.tiny or not args.config:
        config = make_tiny_gpt2_config()
    else:
        raise SystemExit(
            'JSON config loading not wired in Python yet; use --tiny')

    print('=== GPT-2 training (setup) ===')
    print(
        f'hidden={config.hidden_size}  layers={config.num_hidden_layers}  '
        f'heads={config.num_attention_heads}  seq={n_seq}  batch={n_batch}  '
        f'epochs={args.epochs}',
    )

    _ctx = Context(1, 0, 0, '/tmp/nntile_ooc', 16777216, 0, 'localhost', 5001, 0)

    graph = NNGraph('gpt2_training')
    model = Gpt2Causal(graph, 'model', config)

    input_ids = graph.tensor([n_seq, n_batch], DataType.INT64, False)
    input_ids.set_name('input_ids')
    position_ids = graph.tensor([n_seq, n_batch], DataType.INT64, False)
    position_ids.set_name('position_ids')
    attn_mask = graph.tensor([n_seq, n_seq], DataType.BOOL, False)
    attn_mask.set_name('attn_mask')
    input_ids.mark_input(True)
    position_ids.mark_input(True)
    attn_mask.mark_input(True)

    labels = graph.tensor([n_seq, n_batch], DataType.INT64, False)
    labels.set_name('labels')
    labels.mark_input(True)

    if args.load_weights:
        model.load(args.load_weights)
    else:
        init_random_parameter_hints(model, args.seed)

    optimizer = AdamW(
        graph,
        model,
        args.lr,
        args.beta1,
        args.beta2,
        1e-8,
        args.weight_decay,
    )
    print('Optimizer:', optimizer.repr())

    pos_data = np.zeros(n_seq * n_batch, dtype=np.int64)
    fill_arange_position_ids(pos_data, n_seq, n_batch)
    mask_data = sdpa_causal_mask_bool_fortran_fill(n_seq)

    # Phase suffixes keep tensor names unique across incremental steps (C++ default).
    graph.enable_auto_tensor_name_phase_suffix(True)
    ce_scale = 1.0 / float(n_seq * n_batch)

    bound_optimizer_state = False
    train_mmap = TokenMemoryMap(args.train_bin)
    lcfg = CausalLmBatchConfig()
    lcfg.n_seq = n_seq
    lcfg.n_batch = n_batch
    lcfg.shuffle = args.shuffle
    lcfg.seed = args.seed
    mmap_batch = CausalLmBatch()
    train_step = 0

    for epoch in range(args.epochs):
        if args.max_batches > 0 and train_step >= args.max_batches:
            break
        train_it = CausalLmBatchIterator(train_mmap, lcfg, config.vocab_size)
        while train_it.next(mmap_batch):
            if args.max_batches > 0 and train_step >= args.max_batches:
                break

            if train_step > 0:
                for p in graph.parameters():
                    if p.grad is not None:
                        nntile.nn.clear(p.grad)

            logits = model.forward(input_ids, position_ids, attn_mask)
            if logits is None:
                raise RuntimeError('model.forward returned null')

            loss_name = f'loss_s{train_step}'
            loss = nntile.nn.cross_entropy(
                logits, labels, 0, ce_scale, -100)
            loss.set_name(loss_name)
            loss.mark_output(True)

            loss_grad_name = loss_name + '_grad'
            loss_grad = graph.get_or_create_grad(loss, loss_grad_name)
            nntile.nn.fill(1.0, loss_grad)
            loss.backward(True)

            step_lr = scheduled_lr(train_step, args)
            optimizer.step(step_lr)

            graph.finish_phase()
            graph.lower_and_compile()
            runtime = graph.runtime()

            if not bound_optimizer_state:
                for _sname, stensor in optimizer.named_state_tensors():
                    n = 1
                    for d in stensor.shape:
                        n *= d
                    zeros = np.zeros(n, dtype=np.float32)
                    runtime.bind_data(stensor, zeros)
                bound_optimizer_state = True

            input_np = np.asarray(mmap_batch.input_ids, dtype=np.int64)
            labels_np = np.asarray(mmap_batch.target_ids, dtype=np.int64)
            runtime.bind_data(input_ids, input_np)
            runtime.bind_data(labels, labels_np)
            runtime.bind_data(position_ids, pos_data)
            runtime.bind_data(attn_mask, mask_data)

            t0 = time.perf_counter()
            runtime.execute()
            runtime.wait()
            us = (time.perf_counter() - t0) * 1e6

            loss_out = runtime.get_output(loss)
            loss_val = float(loss_out[0])
            if args.warmup_steps > 0:
                print(
                    f'Batch {train_step}  lr={step_lr:.6f}  loss={loss_val}  '
                    f'({us:.0f} us)',
                )
            else:
                print(
                    f'Batch {train_step}  loss={loss_val}  ({us:.0f} us)',
                )

            train_step += 1

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        print(f'output-dir set to {out} (save not implemented in v1)')

    return 0


if __name__ == '__main__':
    sys.exit(main())
