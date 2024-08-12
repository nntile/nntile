# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/gpt2_training.py
# GPT2 training example
#
# @version 1.1.0

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim import SGD, Adam, AdamW
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

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
parser.add_argument("--model", default="gpt2")

parser.add_argument(
    "--pretrained", choices=["local", "remote"], default="remote"
)
parser.add_argument("--checkpoint_path", type=str, default="")
parser.add_argument("--config_path", type=str, default="")
parser.add_argument("--save_checkpoint_path", type=str, default=".model")
parser.add_argument(
    "--optimizer", choices=["sgd", "adam", "adamw"], default="adam"
)


parser.add_argument("--model-path", default=".model")
parser.add_argument("--seq-len-tile", type=int, default=1024)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--minibatch-size", type=int, default=1)
parser.add_argument("--minibatch-size-tile", type=int, default=1)
parser.add_argument("--n-embd-tile", type=int, default=384)
parser.add_argument("--n-inner-tile", type=int, default=1536)
parser.add_argument("--n-head-tile", type=int, default=-1)
parser.add_argument(
    "--torch-device",
    choices=["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4"],
    default="cpu",
)
parser.add_argument(
    "--torch-dtype", choices=["fp32", "fp64", "bf16"], default="fp32"
)
parser.add_argument("--torch-compile", action="store_true")
parser.add_argument(
    "--nntile-dtype", choices=["fp32", "fp64", "tf32", "bf16"], default="fp32"
)
parser.add_argument("--check", action="store_true")
parser.add_argument("--check-fp64", action="store_true")
parser.add_argument("--torch-nforward", type=int, default=0)
parser.add_argument("--torch-nforward-warmup", type=int, default=0)
parser.add_argument("--torch-nbackward", type=int, default=0)
parser.add_argument("--torch-nbackward-warmup", type=int, default=0)
parser.add_argument(
    "--nntile-restrict", choices=["cpu", "cuda", None], default=None
)
parser.add_argument("--nntile-flashattention", action="store_true")
parser.add_argument("--nntile-use-redux", action="store_true")
parser.add_argument("--nntile-nforward", type=int, default=0)
parser.add_argument("--nntile-nforward-warmup", type=int, default=0)
parser.add_argument("--nntile-nbackward", type=int, default=0)
parser.add_argument("--nntile-nbackward-warmup", type=int, default=0)
parser.add_argument("--dataset", default="WikiText-103")
parser.add_argument("--dataset-path", default=".data")
parser.add_argument("--dataset-select", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.0)
parser.add_argument("--torch-nepochs", type=int, default=0)
parser.add_argument("--torch-nepochs-warmup", type=int, default=0)
parser.add_argument("--nntile-nepochs", type=int, default=0)
parser.add_argument("--nntile-nepochs-warmup", type=int, default=0)
parser.add_argument("--nntile-logger", action="store_true")
parser.add_argument(
    "--nntile-logger-server-addr", type=str, default="localhost"
)
parser.add_argument("--nntile-logger-server-port", type=int, default=5001)

# Parse arguments
args = parser.parse_args()
print(args)

# Check arguments
assert args.seq_len_tile > 0
assert args.batch_size > 0
assert args.minibatch_size > 0
assert args.minibatch_size_tile > 0
assert args.batch_size % args.minibatch_size == 0
num_minibatch = args.batch_size // args.minibatch_size
assert args.minibatch_size % args.minibatch_size_tile == 0
assert args.n_embd_tile > 0
assert args.n_inner_tile > 0
assert args.torch_nforward >= 0
assert args.torch_nbackward >= 0
assert args.torch_nepochs >= 0
assert args.nntile_nforward >= 0
assert args.nntile_nbackward >= 0
assert args.nntile_nepochs >= 0

# Set Torch default device to cpu
torch.set_default_device("cpu")

if args.torch_dtype == "fp32":
    torch_dtype = torch.float32
elif args.torch_dtype == "fp64":
    torch_dtype = torch.float64
elif args.torch_dtype == "bf16":
    torch_dtype = torch.bfloat16

if args.nntile_dtype == "tf32":
    torch.backends.cuda.matmul.allow_tf32 = True

# Load named pretrained PyTorch model
if args.pretrained == "remote":
    # Newer versions of transformers can use fast attention, so we disable it
    # through a parameter attn_implementation
    try:
        model_torch = GPT2LMHeadModel.from_pretrained(
            args.model, cache_dir=args.model_path, attn_implementation="eager"
        )
    except Exception:
        model_torch = GPT2LMHeadModel.from_pretrained(
            args.model, cache_dir=args.model_path
        )
elif args.pretrained == "local":
    if args.config_path:
        f = open(args.config_path)
        conf_dict = json.load(f)
        f.close()
        config = GPT2Config(**conf_dict)
        model_torch = GPT2LMHeadModel(config).to(torch_dtype)
        if args.optimizer == "adam":
            optimizer = Adam(model_torch.parameters(), args.lr)
        elif args.optimizer == "sgd":
            optimizer = SGD(model_torch.parameters(), args.lr)
        elif args.optimizer == "adamw":
            optimizer = AdamW(model_torch.parameters(), args.lr)
        else:
            raise ValueError
        if args.checkpoint_path:
            checkpoint = torch.load(args.checkpoint_path)
            model_torch.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

model_torch.eval()
# print(model_torch)

# Create a new PyTorch model with adjusted config and load weights from the
# pretrained one. This step is requried as some operations of GPT2 are still
# pending in NNTile implementation (bias in Linear layers and entire Attention
# layers).
config = model_torch.config
if args.n_head_tile == -1:
    args.n_head_tile = config.n_head
assert config.n_head % args.n_head_tile == 0
assert config.n_positions % args.seq_len_tile == 0
config.attn_pdrop = 0
config.embd_pdrop = 0
config.resid_pdrop = 0
# Current version splits lm_head and wte parameters, shared parameters will be
# supported soon
model_torch.lm_head.weight = nn.Parameter(
    model_torch.lm_head.weight.detach().clone()
)

inner_dim = (
    config.n_inner if config.n_inner is not None else 4 * config.hidden_size
)
config.n_inner = inner_dim

# Print altered PyTorch model to be tested
# print("PyTorch model:")
# print(model_torch)

# Forward FLOPs for Torch without FlashAttention
# Forward FLOPs of matrix products per input sequence per GPT block
torch_nflops_seq_block = (
    2 * config.n_positions * config.n_embd * (3 + 1) * config.n_embd
    + 4 * config.n_positions * config.n_positions * config.n_embd
    + 4 * config.n_positions * config.n_embd * config.n_inner
)
# Forward FLOPs of matrix products per input sequence per GPT model
torch_nflops_seq = (
    config.num_hidden_layers * torch_nflops_seq_block
    + 2 * config.n_positions * config.n_embd * config.vocab_size
)

# FLOPs counting
# MLP
nflops_seq_block_fwd = 4 * config.n_positions * config.n_embd * config.n_inner
nflops_seq_block_bwd = 8 * config.n_positions * config.n_embd * config.n_inner
# Attention Q, K, V
nflops_seq_block_fwd += 8 * config.n_positions * config.n_embd**2
nflops_seq_block_bwd += 16 * config.n_positions * config.n_embd**2
# Attention softmax(Q'@K)@V
if args.nntile_flashattention:
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
    -1,
    -1,
    1,
    args.nntile_logger,
    args.nntile_logger_server_addr,
    args.nntile_logger_server_port,
)
nntile.starpu.profiling_init()
nntile.starpu.profiling_disable()
nntile.starpu.init()
# Restrict computations to CUDA if possible
if args.nntile_restrict == "cuda":
    nntile.starpu.restrict_cuda()
elif args.nntile_restrict == "cpu":
    nntile.starpu.restrict_cpu()
time1 = time.time() - time0
print("StarPU + NNTile + MPI init in {} seconds".format(time1))
next_tag = 0

# optim = torch.optim.SGD(model_torch.parameters(), lr=1e-1)
# input_value = torch.randint(config.vocab_size, \
#             (args.minibatch_size, config.n_positions), dtype=torch.int64)
# output = model_torch(input_value)
# val = torch.mean(output.logits)
# val.backward()
# optim.step()


# Prepare GPT2 model based on the NNTile backend
nntile_model_config = GPT2Config_nntile(
    config.vocab_size,
    args.n_embd_tile,
    config.n_embd,
    args.n_embd_tile,
    config.max_position_embeddings,
    config.n_inner,
    args.n_inner_tile,
    config.layer_norm_epsilon,
    config.num_hidden_layers,
    config.n_head,
    args.n_head_tile,
    "gelutanh",
    args.nntile_flashattention,
    args.nntile_use_redux,
    args.nntile_dtype,
)
nntile_model, next_tag = GPT2Model_nntile.from_torch(
    model_torch,
    args.minibatch_size,
    args.minibatch_size_tile,
    config.n_positions,
    args.seq_len_tile,
    nntile_model_config,
    next_tag,
)

# Move model to the designated device or delete model if it will not be used
# any more
if (
    args.check
    or args.check_fp64
    or args.torch_nforward > 0
    or args.torch_nbackward > 0
    or args.torch_nepochs
):
    model_torch = model_torch.to(args.torch_device)
    if args.torch_compile:
        model_torch = torch.compile(model_torch)
else:
    del model_torch


# Function to check correctness of gradients
def check_grads(model_torch, nntile_model):
    nntile_par_idx = 0
    for name, p_torch in model_torch.named_parameters():
        p_torch_grad_np = p_torch.grad.float().cpu().detach().numpy()
        layer_name = name.split(".")[-2]
        if layer_name == "lm_head":
            p_nntile = nntile_model.parameters[nntile_par_idx]
            p_nntile_grad_np = np.zeros(
                p_nntile.grad.shape, order="F", dtype=np.float32
            )
            p_nntile.grad.to_array(p_nntile_grad_np)
            diff = np.linalg.norm(p_torch_grad_np - p_nntile_grad_np)
            norm = np.linalg.norm(p_torch_grad_np)
            nntile_par_idx += 1
        elif layer_name == "c_attn" and name.split(".")[-1] == "weight":
            diff = 0
            norm = np.linalg.norm(p_torch_grad_np)
            for i_tensor in range(3):
                p_nntile = nntile_model.parameters[nntile_par_idx]
                p_nntile_grad_np = np.zeros(
                    p_nntile.grad.shape, order="F", dtype=np.float32
                )
                p_nntile.grad.to_array(p_nntile_grad_np)
                p_nntile_grad_np = p_nntile_grad_np.reshape(-1, config.n_embd)
                current_grad_block = p_torch_grad_np[
                    :,
                    i_tensor * config.n_embd : (i_tensor + 1) * config.n_embd,
                ]
                diff += (
                    np.linalg.norm(current_grad_block.T - p_nntile_grad_np)
                    ** 2
                )
                nntile_par_idx += 1
            diff = diff**0.5
        elif layer_name == "c_attn" and name.split(".")[-1] == "bias":
            diff = 0
            norm = np.linalg.norm(p_torch_grad_np)
            for i_tensor in range(3):
                p_nntile = nntile_model.parameters[nntile_par_idx]
                p_nntile_grad_np = np.zeros(
                    p_nntile.grad.shape, order="F", dtype=np.float32
                )
                p_nntile.grad.to_array(p_nntile_grad_np)
                p_nntile_grad_np = p_nntile_grad_np.transpose().reshape(-1)
                current_grad_block = p_torch_grad_np[
                    i_tensor * config.n_embd : (i_tensor + 1) * config.n_embd
                ]
                diff += (
                    np.linalg.norm(current_grad_block - p_nntile_grad_np) ** 2
                )
                nntile_par_idx += 1
            diff = diff**0.5
        elif layer_name == "c_proj" and name.split(".")[-3] == "attn":
            if name.split(".")[-1] == "weight":
                diff = 0
                norm = np.linalg.norm(p_torch_grad_np)
                p_nntile = nntile_model.parameters[nntile_par_idx]
                p_nntile_grad_np = np.zeros(
                    p_nntile.grad.shape, order="F", dtype=np.float32
                )
                p_nntile.grad.to_array(p_nntile_grad_np)
                p_nntile_grad_np = p_nntile_grad_np.reshape(config.n_embd, -1)
                diff = np.linalg.norm(p_torch_grad_np.T - p_nntile_grad_np)
                nntile_par_idx += 1
            elif name.split(".")[-1] == "bias":
                norm = np.linalg.norm(p_torch_grad_np)
                p_nntile = nntile_model.parameters[nntile_par_idx]
                p_nntile_grad_np = np.zeros(
                    p_nntile.grad.shape, order="F", dtype=np.float32
                )
                p_nntile.grad.to_array(p_nntile_grad_np)
                diff = np.linalg.norm(p_torch_grad_np - p_nntile_grad_np)
                nntile_par_idx += 1
        elif len(p_torch.shape) == 2:
            p_nntile = nntile_model.parameters[nntile_par_idx]
            p_nntile_grad_np = np.zeros(
                p_nntile.grad.shape, order="F", dtype=np.float32
            )
            p_nntile.grad.to_array(p_nntile_grad_np)
            diff = np.linalg.norm(p_torch_grad_np - p_nntile_grad_np.T)
            norm = np.linalg.norm(p_torch_grad_np)
            nntile_par_idx += 1
        elif len(p_torch.shape) == 1:
            p_nntile = nntile_model.parameters[nntile_par_idx]
            p_nntile_grad_np = np.zeros(
                p_nntile.grad.shape, order="F", dtype=np.float32
            )
            p_nntile.grad.to_array(p_nntile_grad_np)
            diff = np.linalg.norm(p_torch_grad_np - p_nntile_grad_np)
            norm = np.linalg.norm(p_torch_grad_np)
            nntile_par_idx += 1
        print(
            "Gradient of {}: norm={} rel_err={}".format(
                name, norm, diff / norm
            )
        )


# Check accuracy of output and gradients of parmeters if required
if args.check:
    nntile.starpu.pause()
    # Get output from a random input through the forward pass
    input_value = torch.randint(
        config.vocab_size,
        (args.minibatch_size, config.n_positions),
        dtype=torch.int64,
        device=args.torch_device,
    )
    input_value[10:] = config.eos_token_id
    output_value = model_torch.to(torch_dtype)(input_value)
    if args.torch_device.startswith("cuda"):
        torch.cuda.synchronize()
    output_value_np = output_value.logits.float().cpu().detach().numpy()
    # Get gradients of parameters through the backward pass
    loss = 0.5 * (output_value.logits * output_value.logits).sum()
    model_torch.zero_grad()
    loss.backward()
    if args.torch_device.startswith("cuda"):
        torch.cuda.synchronize()
    # Check accuracy of the forward pass by the output activation
    nntile.starpu.resume()
    nntile_model.activations[0].value.from_array(input_value.cpu().numpy().T)
    nntile_model.forward_async()
    nntile.starpu.wait_for_all()
    nntile_output_np = np.zeros_like(output_value_np.T, order="F")
    nntile_model.activations[-1].value.to_array(nntile_output_np)
    diff = np.linalg.norm(nntile_output_np.T - output_value_np)
    norm = np.linalg.norm(output_value_np)
    print("NNTile forward pass relative accuracy: {}".format(diff / norm))
    print("Model output norm: {}".format(norm))
    # Run backward pass by the NNTile to get gradients of parameters
    nntile_model.clear_gradients()
    nntile.tensor.copy_async(
        nntile_model.activations[-1].value, nntile_model.activations[-1].grad
    )
    nntile_model.backward_async()
    nntile.starpu.wait_for_all()
    # Now compare gradients
    check_grads(model_torch, nntile_model)

# Measure throughput of Torch forward pass
if args.torch_nforward > 0:
    input_value = torch.randint(
        config.vocab_size,
        (args.minibatch_size, config.n_positions),
        dtype=torch.int64,
        device=args.torch_device,
    )
    for i in range(args.torch_nforward_warmup):
        output_value = model_torch.to(torch_dtype)(input_value)
    if args.torch_device.startswith("cuda"):
        torch.cuda.synchronize()
    time0 = time.time()
    for i in range(args.torch_nforward):
        output_value = model_torch.to(torch_dtype)(input_value)
    if args.torch_device.startswith("cuda"):
        torch.cuda.synchronize()
    time1 = time.time() - time0
    print(
        "Torch forward throughput (sequence/sec): ",
        args.torch_nforward * args.minibatch_size / time1,
    )
    print(
        "Torch forward performance: {} Tflops/s".format(
            torch_nflops_seq
            * args.torch_nforward
            * args.minibatch_size
            / time1
            * 1e-12
        )
    )

# Measure throughput of Torch backward pass
if args.torch_nbackward > 0:
    input_value = torch.randint(
        config.vocab_size,
        (args.minibatch_size, config.n_positions),
        dtype=torch.int64,
        device=args.torch_device,
    )
    output_value = model_torch.to(torch_dtype)(input_value)
    loss = (output_value.logits * output_value.logits).sum()
    for i in range(args.torch_nbackward_warmup):
        loss.backward(retain_graph=True)
    if args.torch_device.startswith("cuda"):
        torch.cuda.synchronize()
    time0 = time.time()
    for i in range(args.torch_nbackward):
        loss.backward(retain_graph=True)
    if args.torch_device.startswith("cuda"):
        torch.cuda.synchronize()
    time1 = time.time() - time0
    print(
        "Torch backward throughput (sequence/sec): ",
        args.torch_nbackward * args.minibatch_size / time1,
    )
    print(
        "Torch backward performance: {} Tflops/s".format(
            2
            * torch_nflops_seq
            * args.torch_nbackward
            * args.minibatch_size
            / time1
            * 1e-12
        )
    )

# Measure throughput of the forward pass by NNTile
if args.nntile_nforward > 0:
    input_value = torch.randint(
        config.vocab_size,
        (args.minibatch_size, config.n_positions),
        dtype=torch.int64,
    )
    nntile_model.activations[0].value.from_array(input_value.T)
    for i in range(args.nntile_nforward_warmup):
        nntile_model.forward_async()
    nntile.starpu.wait_for_all()
    time0 = time.time()
    for i in range(args.nntile_nforward):
        nntile_model.forward_async()
    nntile.starpu.wait_for_all()
    time1 = time.time() - time0
    print(
        "NNTile forward throughput (sequence/sec): ",
        args.nntile_nforward * args.minibatch_size / time1,
    )
    print(
        "NNTile forward performance: {} Tflops/s".format(
            nflops_seq_fwd
            * args.nntile_nforward
            * args.minibatch_size
            / time1
            * 1e-12
        )
    )

# Measure throughput of the forward pass by NNTile
if args.nntile_nbackward > 0:
    input_value = torch.randint(
        config.vocab_size,
        (args.minibatch_size, config.n_positions),
        dtype=torch.int64,
    )
    nntile_model.activations[0].value.from_array(input_value.T)
    nntile_model.clear_gradients()
    for i in range(args.nntile_nbackward_warmup):
        nntile_model.forward_async()
        nntile.tensor.copy_async(
            nntile_model.activations[-1].value,
            nntile_model.activations[-1].grad,
        )
        nntile_model.backward_async()
    nntile.starpu.wait_for_all()
    time0 = time.time()
    nntile_model.clear_gradients()
    for i in range(args.nntile_nbackward):
        nntile_model.forward_async()
        nntile.tensor.copy_async(
            nntile_model.activations[-1].value,
            nntile_model.activations[-1].grad,
        )
        nntile_model.backward_async()
    nntile.starpu.wait_for_all()
    time1 = time.time() - time0
    print(
        "NNTile forward+backward throughput (sequence/sec): ",
        args.nntile_nbackward * args.minibatch_size / time1,
    )
    print(
        "NNTile forward+backward performance: {} Tflops/s".format(
            nflops_seq
            * args.nntile_nbackward
            * args.minibatch_size
            / time1
            * 1e-12
        )
    )

# Prepare input and output batches if real training is required
if args.torch_nepochs > 0 or args.nntile_nepochs > 0 or args.check_fp64:
    # Read dataset
    if args.dataset == "WikiText-103":
        train_dataset = load_dataset(
            "wikitext",
            "wikitext-103-v1",
            split="train",
            cache_dir=args.dataset_path,
        ).select(np.arange(args.dataset_select, dtype=np.int64))
        test_dataset = load_dataset(
            "wikitext",
            "wikitext-103-v1",
            split="test",
            cache_dir=args.dataset_path,
        )
    else:
        raise ValueError(
            "{} dataset is not supported yet!".format(args.dataset)
        )
    # Tokenize and store as a single numpy array
    tokenizer = GPT2TokenizerFast.from_pretrained(
        args.model, cache_dir=args.model_path
    )
    map_train_tokens = map(
        lambda x: tokenizer(x["text"])["input_ids"], train_dataset
    )
    list_train_tokens = []
    for seq in map_train_tokens:
        list_train_tokens.extend(seq)
    num_train_tokens = len(list_train_tokens)
    num_train_seq = num_train_tokens // (config.n_positions + 1)
    num_train_batches = num_train_seq // args.batch_size
    num_train_tokens_truncated = (
        num_train_batches * args.batch_size * (config.n_positions + 1)
    )
    train_tokens = np.array(
        list_train_tokens[:num_train_tokens_truncated],
        order="F",
        dtype=np.int64,
    )
    train_tokens = train_tokens.reshape(
        num_train_batches,
        num_minibatch,
        args.minibatch_size,
        config.n_positions + 1,
    )
    print(
        "Number of train sequences: {}".format(
            num_train_batches * args.batch_size
        )
    )
    print("Number of train batches: {}".format(num_train_batches))

# Check accuracy of output and gradients of parmeters if required for float64
# type on the torch side
if args.check_fp64:
    model64_torch = model_torch.to(torch.float64)
    # Get output from a random input through the forward pass
    # input_value = torch.randint(config.vocab_size, \
    #        (num_minibatch, args.minibatch_size, config.n_positions), \
    #        dtype=torch.int64, device=args.torch_device)
    input_value = torch.tensor(train_tokens[5, 0, :, :-1]).to(
        args.torch_device
    )
    output_label = torch.tensor(train_tokens[5, 0, :, 1:]).to(
        args.torch_device
    )
    output_value = model64_torch(input_value)
    if args.torch_device.startswith("cuda"):
        torch.cuda.synchronize()
    output_value_np = output_value.logits.cpu().detach().numpy()
    # Get gradients of parameters through the backward pass
    # loss = 0.5 * (output_value.logits * output_value.logits).sum()
    # loss.backward()
    loss_func = nn.CrossEntropyLoss(reduction="mean")
    output_logits = output_value.logits.reshape(-1, config.vocab_size)
    loss = loss_func(output_logits, output_label.reshape(-1))
    model_torch.zero_grad()
    loss.backward()
    if args.torch_device.startswith("cuda"):
        torch.cuda.synchronize()
    # Check accuracy of the forward pass by the output activation
    nntile_model.activations[0].value.from_array(input_value.cpu().numpy().T)
    nntile_model.forward_async()
    nntile.starpu.wait_for_all()
    nntile_output_np = np.zeros_like(
        output_value_np.T, order="F", dtype=np.float32
    )
    nntile_model.activations[-1].value.to_array(nntile_output_np)
    diff = np.linalg.norm(nntile_output_np.T - output_value_np)
    norm = np.linalg.norm(output_value_np)
    print("NNTile forward pass relative accuracy: {}".format(diff / norm))
    print("Model output norm: {}".format(norm))
    # Run backward pass by the NNTile to get gradients of parameters
    loss, next_tag = nntile.loss.CrossEntropy.generate_simple(
        nntile_model.activations[-1],
        next_tag,
        scale=1.0 / (args.batch_size * config.n_positions),
    )
    loss.y.from_array(train_tokens[5, 0, :, 1:].T)
    nntile_model.clear_gradients()
    loss.calc_async()
    nntile_model.backward_async()
    nntile.starpu.wait_for_all()
    # Now compare gradients
    check_grads(model64_torch, nntile_model)
    loss.unregister()

# Train neural network by the NNTile
if args.nntile_nepochs > 0:
    # Prepare input and output batches for training by NNTile
    time0 = time.time()
    batch_input = []
    batch_output = []
    x_traits = nntile.tensor.TensorTraits(
        [config.n_positions, args.minibatch_size],
        [args.seq_len_tile, args.minibatch_size_tile],
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
    print("From PyTorch loader to NNTile batches in {} seconds".format(time1))
    # Set up learning rate and optimizer for training
    if args.optimizer == "adam":
        optimizer = nntile.optimizer.Adam(
            nntile_model.get_parameters(), args.lr, next_tag
        )
    elif args.optimizer == "adamw":
        optimizer = nntile.optimizer.AdamW(
            nntile_model.get_parameters(), args.lr, next_tag
        )
    elif args.optimizer == "sgd":
        optimizer = nntile.optimizer.SGD(
            nntile_model.get_parameters(), args.lr, next_tag
        )
    next_tag = optimizer.get_next_tag()
    # Define Cross Entropy loss function
    loss, next_tag = nntile.loss.CrossEntropy.generate_simple(
        nntile_model.activations[-1],
        next_tag,
        scale=1.0 / (args.batch_size * config.n_positions),
    )
    # Set up training pipeline
    pipeline = nntile.pipeline.Pipeline(
        batch_input,
        batch_output,
        nntile_model,
        optimizer,
        loss,
        args.nntile_nepochs_warmup,
    )
    # Warmup training
    # nntile.starpu.pause()
    pipeline.train_async()
    # nntile.starpu.resume()
    nntile.starpu.wait_for_all()
    # Actual training
    pipeline.n_epochs = args.nntile_nepochs
    nntile.starpu.profiling_enable()
    # nntile.starpu.pause()
    time0 = time.time()
    pipeline.train_async()
    # nntile.starpu.resume()
    nntile.starpu.wait_for_all()
    nntile.starpu.profiling_disable()
    time1 = time.time() - time0
    print("NNTile training time: {} seconds".format(time1))
    print(
        "NNTile training throughput tokens/sec: {}".format(
            args.nntile_nepochs
            * num_train_batches
            * args.batch_size
            * config.n_positions
            / time1
        )
    )
    print(
        "NNTile performance: {} Tflops/s".format(
            nflops_seq
            * args.nntile_nepochs
            * num_train_batches
            * args.batch_size
            / time1
            * 1e-12
        )
    )
    loss_np = np.zeros((1), dtype=np.float32)
    loss.val.to_array(loss_np)
    print("NNTile loss on the last batch: {}".format(loss_np[0]))
    loss.unregister()
    optimizer.unregister()
    for batch in batch_input + batch_output:
        for x in batch:
            x.unregister()

# Unregister all tensors related to model
nntile_model.unregister()

if args.torch_nepochs > 0:
    torch_input = []
    torch_output = []
    for i in range(num_train_batches):
        minibatch_input = []
        minibatch_output = []
        for j in range(num_minibatch):
            minibatch_input.append(
                torch.tensor(
                    train_tokens[i, j, :, :-1], requires_grad=False
                ).contiguous()
            )
            minibatch_output.append(
                torch.tensor(
                    train_tokens[i, j, :, 1:], requires_grad=False
                ).contiguous()
            )
        torch_input.append(minibatch_input)
        torch_output.append(minibatch_output)
    loss_func = nn.CrossEntropyLoss(reduction="mean")
    if args.optimizer == "adam":
        optimizer = Adam(model_torch.parameters(), args.lr)
    elif args.optimizer == "sgd":
        optimizer = SGD(model_torch.parameters(), args.lr)
    elif args.optimizer == "adamw":
        optimizer = AdamW(model_torch.parameters(), args.lr)
    else:
        raise ValueError
    # Warmup training
    for i in range(args.torch_nepochs_warmup):
        for j in range(num_train_batches):
            optimizer.zero_grad()
            loss = torch.zeros(1, dtype=torch_dtype, device=args.torch_device)
            for k in range(num_minibatch):
                train_input = torch_input[j][k].to(args.torch_device)
                train_labels = (
                    torch_output[j][k].to(args.torch_device).reshape(-1)
                )
                train_output = model_torch.to(torch_dtype)(train_input)
                train_logits = train_output.logits.reshape(
                    -1, config.vocab_size
                )
                loss_local = loss_func(train_logits, train_labels)
                loss_local.backward()
                loss += loss_local
            print("loss={}".format(loss.item()), flush=True)
            optimizer.step()
    # Actual training
    if args.torch_device.startswith("cuda"):
        torch.cuda.synchronize()
    time0 = time.time()
    for i in range(args.torch_nepochs):
        for j in range(num_train_batches):
            optimizer.zero_grad()
            loss = torch.zeros(1, dtype=torch_dtype, device=args.torch_device)
            for k in range(num_minibatch):
                train_input = torch_input[j][k].to(args.torch_device)
                train_labels = (
                    torch_output[j][k].to(args.torch_device).reshape(-1)
                )
                train_output = model_torch.to(torch_dtype)(train_input)
                train_logits = train_output.logits.reshape(
                    -1, config.vocab_size
                )
                loss_local = loss_func(train_logits, train_labels)
                loss_local.backward()
                loss += loss_local
            print(
                "Batch={}/{} Epoch={}/{} Loss={}".format(
                    j + 1,
                    num_train_batches,
                    i + 1,
                    args.torch_nepochs,
                    loss.item(),
                ),
                flush=True,
            )
            optimizer.step()
    if args.torch_device.startswith("cuda"):
        torch.cuda.synchronize()
    time1 = time.time() - time0
    print("Torch training time: {} seconds".format(time1), flush=True)
    print(
        "Torch training throughput tokens/sec: {}".format(
            args.torch_nepochs
            * num_train_batches
            * args.batch_size
            * config.n_positions
            / time1
        ),
        flush=True,
    )
    print(
        "Torch performance: {} Tflops/s".format(
            3
            * torch_nflops_seq
            * args.torch_nepochs
            * num_train_batches
            * args.batch_size
            / time1
            * 1e-12
        ),
        flush=True,
    )
    print("Torch loss on the last batch: {}".format(loss.item()), flush=True)
