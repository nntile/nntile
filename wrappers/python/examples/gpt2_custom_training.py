# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/gpt2_custom_training.py
# Custom GPT2 training example
#
# @version 1.1.0

# Import system packages
import argparse
import json
import time

# Import 3rd party packages
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

# Import NNTile
import nntile
from nntile.model.gpt2 import (
    GPT2Config as GPT2Config_nntile, GPT2Model as GPT2Model_nntile)

# Create argument parser
parser = argparse.ArgumentParser(
    prog="GPT2-based neural networks",
    description="This example presents an NNTile implementation of a "
    "GPT2-family of models and compares it against the Huggingface. "
    "It checks relative accuracy of a forward pass (values of "
    "activations) and backward pass (gradients of parameters and "
    "activations) and a throughput of inference and training. It can "
    "also fine-tune a pretrained NNTile model on a chosen dataset.",
)
parser.add_argument("--config-path")
parser.add_argument("--tokenizer", default="gpt2")
parser.add_argument("--tokenizer-path")
parser.add_argument("--load-checkpoint")
parser.add_argument("--load-optimizer")
parser.add_argument("--save-checkpoint")
parser.add_argument(
    "--save-checkpoint-dtype", choices=["fp32", "fp16", "bf16"], default="fp32"
)
parser.add_argument("--save-optimizer")
parser.add_argument(
    "--save-optimizer-dtype", choices=["fp32", "fp16", "bf16"], default="fp32"
)
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
parser.add_argument("--nforward", type=int, default=0)
parser.add_argument("--nforward-warmup", type=int, default=0)
parser.add_argument("--nbackward", type=int, default=0)
parser.add_argument("--nbackward-warmup", type=int, default=0)
parser.add_argument(
    "--dataset", choices=["WikiText-103", "train.bin"], default="WikiText-103"
)
parser.add_argument("--dataset-path", default=".data")
parser.add_argument("--dataset-select", type=int, default=100)
parser.add_argument(
    "--optimizer", choices=["sgd", "adam", "adamw"], default="adamw"
)
parser.add_argument("--optimizer-eps", type=float, default=1e-8)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument(
    "--loss-reduction", choices=["sum", "mean"], default="mean"
)
parser.add_argument("--lr", type=float, default=0.0)
parser.add_argument("--start-lr", type=float, default=0.0)
parser.add_argument("--full-lr-iter", type=int, default=1)
parser.add_argument("--nepochs", type=int, default=0)
parser.add_argument("--nepochs-warmup", type=int, default=0)
parser.add_argument(
    "--dtype", choices=["fp32", "tf32", "bf16", "fp64"], default="fp32"
)
parser.add_argument("--logger", action="store_true")
parser.add_argument("--logger-server-addr", type=str, default="localhost")
parser.add_argument("--logger-server-port", type=int, default=5001)

# Parse arguments
args = parser.parse_args()
print(args, flush=True)

# Set Torch default device to cpu and fix random seed
torch.set_default_device("cpu")
torch.manual_seed(0)

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

# Load model from checkpoint if needed
if args.load_checkpoint is not None:
    checkpoint = torch.load(args.load_checkpoint, map_location="cpu")
    for key in checkpoint["model_state_dict"]:
        checkpoint["model_state_dict"][key] = checkpoint["model_state_dict"][
            key
        ].to(torch.float32)
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
assert args.nforward >= 0
assert args.nforward_warmup >= 0
assert args.nbackward >= 0
assert args.nbackward_warmup >= 0
assert args.nepochs >= 0
assert args.nepochs_warmup >= 0
assert args.weight_decay >= 0

# Print altered PyTorch model to be tested
print("PyTorch model:")
print(model_torch, flush=True)

# FLOPs counting
# MLP
nflops_seq_block_fwd = 4 * config.n_positions * config.n_embd * config.n_inner
nflops_seq_block_bwd = 8 * config.n_positions * config.n_embd * config.n_inner
# Attention Q, K, V
nflops_seq_block_fwd += 8 * config.n_positions * config.n_embd**2
nflops_seq_block_bwd += 16 * config.n_positions * config.n_embd**2
# Attention softmax(Q'@K)@V
if args.flashattention:
    nflops_seq_block_fwd += 6 * config.n_positions**2 * config.n_embd
    nflops_seq_block_bwd += 14 * config.n_positions**2 * config.n_embd
else:
    nflops_seq_block_fwd += 4 * config.n_positions**2 * config.n_embd
    nflops_seq_block_bwd += 8 * config.n_positions**2 * config.n_embd
# Total flops with LM_head
nflops_seq_fwd = (
    config.num_hidden_layers * nflops_seq_block_fwd
    + 2 * config.n_positions * config.n_embd * config.vocab_size
)
nflops_seq_bwd = (
    config.num_hidden_layers * nflops_seq_block_bwd
    + 4 * config.n_positions * config.n_embd * config.vocab_size
)
nflops_seq = nflops_seq_fwd + nflops_seq_bwd

# Initialize NNTile and StarPU
time0 = time.time()
# Set up StarPU+MPI and init codelets
nntile_config = nntile.starpu.Config(
    -1, -1, 1, args.logger, args.logger_server_addr, args.logger_server_port
)
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
    args.dtype,
)
model_nntile, next_tag = GPT2Model_nntile.from_torch(
    model_torch,
    args.minibatch,
    args.minibatch_tile,
    config.n_positions,
    args.seq_tile,
    model_nntile_config,
    next_tag,
)
del model_torch

# Measure throughput of the forward pass by NNTile
if args.nforward > 0:
    input_value = torch.randint(
        config.vocab_size,
        (args.minibatch, config.n_positions),
        dtype=torch.int64,
    )
    model_nntile.activations[0].value.from_array(input_value.T)
    for i in range(args.nforward_warmup):
        model_nntile.forward_async()
    nntile.starpu.wait_for_all()
    time0 = time.time()
    for i in range(args.nforward):
        model_nntile.forward_async()
    nntile.starpu.wait_for_all()
    time1 = time.time() - time0
    print("NNTile forward time: {} seconds".format(time1))
    print(
        "NNTile forward throughput (tokens/sec): ",
        config.n_positions * args.nforward * args.minibatch / time1,
    )
    print(
        "NNTile forward performance: {} Tflops/s".format(
            nflops_seq_fwd * args.nforward * args.minibatch / time1 * 1e-12
        ),
        flush=True,
    )

# Measure throughput of the forward pass by NNTile
if args.nbackward > 0:
    input_value = torch.randint(
        config.vocab_size,
        (args.minibatch, config.n_positions),
        dtype=torch.int64,
    )
    model_nntile.activations[0].value.from_array(input_value.T)
    model_nntile.clear_gradients()
    for i in range(args.nbackward_warmup):
        model_nntile.forward_async()
        nntile.tensor.copy_async(
            model_nntile.activations[-1].value,
            model_nntile.activations[-1].grad,
        )
        model_nntile.backward_async()
    nntile.starpu.wait_for_all()
    time0 = time.time()
    model_nntile.clear_gradients()
    for i in range(args.nbackward):
        model_nntile.forward_async()
        nntile.tensor.copy_async(
            model_nntile.activations[-1].value,
            model_nntile.activations[-1].grad,
        )
        model_nntile.backward_async()
    nntile.starpu.wait_for_all()
    time1 = time.time() - time0
    print("NNTile forward+backward time: {} seconds".format(time1))
    print(
        "NNTile forward+backward throughput (tokens/sec): ",
        config.n_positions * args.nbackward * args.minibatch / time1,
    )
    print(
        "NNTile forward+backward performance: {} Tflops/s".format(
            (nflops_seq_fwd + nflops_seq_bwd)
            * args.nbackward
            * args.minibatch
            / time1
            * 1e-12
        ),
        flush=True,
    )

# Prepare input and output batches if real training is required
# Read dataset
if args.dataset == "WikiText-103":
    train_dataset = load_dataset(
        "wikitext",
        "wikitext-103-v1",
        split="train",
        cache_dir=args.dataset_path,
    )
    if args.dataset_select != -1:
        train_dataset = train_dataset.select(
            np.arange(args.dataset_select, dtype=np.int64)
        )
    test_dataset = load_dataset(
        "wikitext",
        "wikitext-103-v1",
        split="test",
        cache_dir=args.dataset_path,
    )
    tokenized = False
elif args.dataset == "train.bin":
    train_data = np.memmap(args.dataset_path, dtype=np.uint16, mode="r")
    train_tokens = np.array(train_data, order="F", dtype=np.int64)
    tokenized = True
    del train_data
else:
    raise ValueError("{} dataset is not supported yet!".format(args.dataset))
# Tokenize and store as a single numpy array
if not tokenized:
    tokenizer = GPT2TokenizerFast.from_pretrained(
        args.tokenizer, cache_dir=args.tokenizer_path
    )
    map_train_tokens = map(
        lambda x: tokenizer(x["text"])["input_ids"], train_dataset
    )
    list_train_tokens = []
    for seq in map_train_tokens:
        list_train_tokens.extend(seq)
    train_tokens = np.array(list_train_tokens, dtype=np.int64)

num_train_tokens = train_tokens.shape[0]
num_train_seq = num_train_tokens // (config.n_positions + 1)
num_train_batches = num_train_seq // args.batch
num_train_tokens_truncated = (
    num_train_batches * args.batch * (config.n_positions + 1)
)
train_tokens = np.array(
    train_tokens[:num_train_tokens_truncated], order="F", dtype=np.int64
)
train_tokens = train_tokens.reshape(
    num_train_batches, num_minibatch, args.minibatch, config.n_positions + 1
)
print("Total train batches: {}".format(num_train_batches))
print("Total train sequences: {}".format(num_train_batches * args.batch))
print(
    "Total train tokens: {}".format(
        num_train_batches * args.batch * config.n_positions
    ),
    flush=True,
)

# Train neural network by the NNTile
# Prepare input and output batches for training by NNTile
time0 = time.time()
batch_input = []
batch_output = []
x_traits = nntile.tensor.TensorTraits(
    [config.n_positions, args.minibatch], [args.seq_tile, args.minibatch_tile]
)
x_distr = [0] * x_traits.grid.nelems
for i in range(num_train_batches):
    minibatch_input = []
    minibatch_output = []
    for j in range(num_minibatch):
        x = nntile.tensor.Tensor_int64(x_traits, x_distr, next_tag)
        next_tag = x.next_tag
        x.from_array(np.asfortranarray(train_tokens[i, j, :, :-1].T))
        minibatch_input.append(x)
        y = nntile.tensor.Tensor_int64(x_traits, x_distr, next_tag)
        next_tag = y.next_tag
        y.from_array(np.asfortranarray(train_tokens[i, j, :, 1:].T))
        minibatch_output.append(y)
    batch_input.append(minibatch_input)
    batch_output.append(minibatch_output)
time1 = time.time() - time0
print(
    "From PyTorch loader to NNTile batches in {} seconds".format(time1),
    flush=True,
)
# Set up learning rate and optimizer for training
if args.optimizer == "adamw":
    optimizer = nntile.optimizer.AdamW(
        model_nntile.get_parameters(),
        args.lr,
        next_tag,
        eps=args.optimizer_eps,
        start_lr=args.start_lr,
        full_lr_iter=args.full_lr_iter,
        weight_decay=args.weight_decay,
    )
elif args.optimizer == "adam":
    optimizer = nntile.optimizer.Adam(
        model_nntile.get_parameters(),
        args.lr,
        next_tag,
        eps=args.optimizer_eps,
        weight_decay=args.weight_decay,
    )
elif args.optimizer == "sgd":
    optimizer = nntile.optimizer.SGD(
        model_nntile.get_parameters(),
        args.lr,
        next_tag,
        weight_decay=args.weight_decay,
    )
next_tag = optimizer.get_next_tag()
# Load optimizer state if needed
if args.load_optimizer is not None:
    optimizer.load_state(args.load_optimizer)
    optimizer.lr = args.lr
    optimizer.start_lr = args.start_lr
    optimizer.full_lr_iter = args.full_lr_iter
# Define Cross Entropy loss function
if args.loss_reduction == "sum":
    loss_scale = 1.0
elif args.loss_reduction == "mean":
    loss_scale = 1.0 / (args.batch * config.n_positions)
loss, next_tag = nntile.loss.CrossEntropy.generate_simple(
    model_nntile.activations[-1], next_tag, scale=loss_scale
)
# Set up training pipeline
pipeline = nntile.pipeline.Pipeline(
    batch_input,
    batch_output,
    model_nntile,
    optimizer,
    loss,
    args.nepochs_warmup,
)
# Warmup training
pipeline.train_async()
nntile.starpu.wait_for_all()
# Actual training
pipeline.n_epochs = args.nepochs
nntile.starpu.profiling_enable()
# nntile.starpu.pause()
time0 = time.time()
pipeline.train_async()
# nntile.starpu.resume()
nntile.starpu.wait_for_all()
nntile.starpu.profiling_disable()
time1 = time.time() - time0
print("Training time: {} seconds".format(time1))
print(
    "Training throughput tokens/sec: {}".format(
        args.nepochs
        * num_train_batches
        * args.batch
        * config.n_positions
        / time1
    )
)
print(
    "Training performance: {} Tflops/s".format(
        nflops_seq
        * args.nepochs
        * num_train_batches
        * args.batch
        / time1
        * 1e-12
    )
)
loss_np = np.zeros((1), dtype=np.float32)
loss.val.to_array(loss_np)
print("Loss on the last batch: {}".format(loss_np[0]), flush=True)

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

# Save model into checkpoint if needed
if args.save_checkpoint is not None:
    model_torch = GPT2LMHeadModel(config)
    model_torch.lm_head.weight = nn.Parameter(
        model_torch.lm_head.weight.detach().clone()
    )
    model_nntile.to_torch(model_torch)
    state_dict = model_torch.state_dict()
    # Quantize if needed
    if args.save_checkpoint_dtype == "fp16":
        for key in state_dict:
            state_dict[key] = state_dict[key].to(torch.float16)
    elif args.save_checkpoint_dtype == "bf16":
        for key in state_dict:
            state_dict[key] = state_dict[key].to(torch.bfloat16)
    torch.save({"model_state_dict": state_dict}, args.save_checkpoint)
    del model_torch, state_dict

# Save optimizer state if needed
if args.save_optimizer is not None:
    optimizer.save_state(args.save_optimizer, dtype=args.save_optimizer_dtype)

# Unregister all StarPU buffers
loss.unregister()
optimizer.unregister()
for batch in batch_input + batch_output:
    for x in batch:
        x.unregister()

# Unregister all tensors related to model, that are still registered
model_nntile.unregister()
