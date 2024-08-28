# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/gpt2_custom_generate.py
# GPT2 generate example
#
# @version 1.1.0

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

import nntile
from nntile.model.gpt2 import (
    GPT2Config as GPT2Config_nntile, GPT2Model as GPT2Model_nntile)

# Create argument parser
parser = argparse.ArgumentParser(
    prog="GPT2-based neural networks",
    description="This example presents an NNTile implementation of a "
    "GPT2-family of models and provides functionality for text "
    "generation.",
)
parser.add_argument("--config-path")
parser.add_argument("--tokenizer", default="gpt2")
parser.add_argument("--tokenizer-path")
parser.add_argument("--load-checkpoint")
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--minibatch", type=int, default=-1)
parser.add_argument("--minibatch-tile", type=int, default=-1)
parser.add_argument("--seq-tile", type=int, default=-1)
parser.add_argument("--embd-tile", type=int, default=-1)
parser.add_argument("--inner-tile", type=int, default=-1)
parser.add_argument("--head-tile", type=int, default=-1)
parser.add_argument("--restrict", choices=["cpu", "cuda", None], default=None)
parser.add_argument("--flashattention", action="store_true")
parser.add_argument("--redux", action="store_true")
parser.add_argument("--fp32-fast-tf32", action="store_true")
parser.add_argument("--nwarmup", type=int, default=0)
parser.add_argument("--input", choices=["text"], default="text")
parser.add_argument("--input-path", default="input.txt")
parser.add_argument("--ntokens", type=int, default=10)

# Parse arguments
args = parser.parse_args()
print(args, flush=True)

# Set Torch default device to cpu
torch.set_default_device("cpu")

# Create model from config and disconnect embedding and lm head
with open(args.config_path, "r") as fd:
    conf_dict = json.load(fd)
config = GPT2Config(**conf_dict)
config.n_inner = 4 * config.n_embd
model_torch = GPT2LMHeadModel(config)
model_torch.lm_head.weight = nn.Parameter(
    model_torch.lm_head.weight.detach().clone()
)

# Disable dropout, as it is not supported by NNTile yet
config.pdrop = 0
config.attn_pdrop = 0
config.embd_pdrop = 0
config.resid_pdrop = 0

# Load model from checkpoint
checkpoint = torch.load(args.load_checkpoint, map_location="cpu")
model_torch.load_state_dict(checkpoint["model_state_dict"])
del checkpoint

# Check sizes
assert args.batch > 0
if args.minibatch == -1:
    args.minibatch = args.batch
assert args.minibatch > 0
assert args.batch % args.minibatch == 0
num_minibatch = args.batch // args.minibatch
if args.minibatch_tile == -1:
    args.minibatch_tile = args.minibatch
assert args.minibatch_tile > 0
if args.seq_tile == -1:
    args.seq_tile = config.n_positions
assert args.seq_tile > 0
assert config.n_positions % args.seq_tile == 0
if args.embd_tile == -1:
    args.embd_tile = config.n_embd
assert args.embd_tile > 0
if args.inner_tile == -1:
    args.inner_tile = config.n_inner
assert args.inner_tile > 0
if args.head_tile == -1:
    args.head_tile = config.n_head
assert args.head_tile > 0
assert config.n_head % args.head_tile == 0
assert args.nwarmup >= 0

# Print altered PyTorch model to be tested
print("PyTorch model:")
print(model_torch, flush=True)

# FLOPs counting
# MLP
nflops_seq_block_fwd = 4 * config.n_positions * config.n_embd * config.n_inner
# Attention Q, K, V
nflops_seq_block_fwd += 8 * config.n_positions * config.n_embd**2
# Attention softmax(Q'@K)@V
if args.flashattention:
    nflops_seq_block_fwd += 6 * config.n_positions**2 * config.n_embd
else:
    nflops_seq_block_fwd += 4 * config.n_positions**2 * config.n_embd
# Total flops with LM_head
nflops_seq_fwd = (
    config.num_hidden_layers * nflops_seq_block_fwd
    + 2 * config.n_positions * config.n_embd * config.vocab_size
)
nflops_seq = nflops_seq_fwd

# Initialize NNTile and StarPU
time0 = time.time()
# Set up StarPU+MPI and init codelets
nntile_config = nntile.starpu.Config(-1, -1, 1)
nntile.starpu.profiling_init()
nntile.starpu.profiling_disable()
nntile.starpu.init()
# Restrict computations to CUDA if possible
if args.restrict == "cuda":
    nntile.starpu.restrict_cuda()
elif args.restrict == "cpu":
    nntile.starpu.restrict_cpu()
time1 = time.time() - time0
print("StarPU + NNTile + MPI init in {} seconds".format(time1), flush=True)
next_tag = 0

# Prepare GPT2 model based on the NNTile backend
model_nntile_config = GPT2Config_nntile(
    config.vocab_size,
    args.embd_tile,
    config.n_embd,
    args.embd_tile,
    config.max_position_embeddings,
    config.n_inner,
    args.inner_tile,
    config.layer_norm_epsilon,
    config.num_hidden_layers,
    config.n_head,
    args.head_tile,
    "gelutanh",
    args.flashattention,
    args.redux,
)
model_nntile, next_tag = GPT2Model_nntile.from_torch(
    model_torch,
    args.minibatch,
    args.minibatch_tile,
    config.n_positions,
    args.seq_tile,
    model_nntile_config,
    next_tag,
    args.fp32_fast_tf32,
)
# model_torch.eval()
del model_torch

# Warmup
if args.nwarmup > 0:
    input_value = torch.randint(
        config.vocab_size, (1, config.n_positions), dtype=torch.int64
    )
    model_nntile.activations[0].value.from_array(input_value.T)
    for i in range(args.nwarmup):
        model_nntile.forward_async()
    nntile.starpu.wait_for_all()

# Prepare input batches
if args.input == "text":
    with open(args.input_path) as fd:
        lines = fd.readlines()
    tokenizer = GPT2TokenizerFast.from_pretrained(
        args.tokenizer, cache_dir=args.tokenizer_path
    )
    input_numpy = config.eos_token_id * np.ones(
        (1, config.n_positions), dtype=np.int64
    )
    input_tokens = np.array(
        list(map(lambda x: tokenizer(x)["input_ids"], lines))
    )
    input_tokens_start = input_tokens.shape[1] - 1
    input_numpy[0, 0:input_tokens_start] = input_tokens[0, :-1]

# Run forward 50 times autoregressively
output_numpy = np.zeros(
    (config.vocab_size, config.n_positions, 1), dtype=np.float32, order="F"
)
for i in range(args.ntokens):
    model_nntile.activations[0].value.from_array(input_numpy.T)
    model_nntile.forward_async()
    model_nntile.activations[-1].value.to_array(output_numpy)
    new_id = output_numpy[:50257, input_tokens_start + i - 1, 0].argmax()
    input_numpy[0, input_tokens_start + i] = new_id
    print(tokenizer.decode(input_numpy[0, 0 : input_tokens_start + i + 1]))

nntile.starpu.wait_for_all()
time1 = time.time() - time0
print("Generate time: {} seconds".format(time1))
print(
    "Generate throughput tokens/sec: {}".format(
        args.ntokens * config.n_positions / time1
    )
)
print(
    "Generate performance: {} Tflops/s".format(
        nflops_seq * args.ntokens / time1 * 1e-12
    )
)

# Unregister intermediate activations to free some space
for t in model_nntile.activations:
    t.unregister()

# Unregister gradients of parameters to free some space
for t in model_nntile.parameters:
    if t.grad is not None and t.grad_required:
        t.grad.unregister()

# Unregister temporaries of each layer to free some space
for layer in model_nntile.layers:
    for t in layer.temporaries:
        if t is not None:
            t.unregister()

# Unregister all tensors related to model, that are still registered
model_nntile.unregister()
