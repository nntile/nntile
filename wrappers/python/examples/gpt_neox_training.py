# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/gpt_neox_training.py
# GPTNeoXForCausalLM training example
#
# @version 1.1.0

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import SGD, Adam, AdamW
from transformers import GPTNeoXConfig as ConfigTorch
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXForCausalLM as ModelTorch)

import nntile
from nntile.model.gpt_neox_causal import GPTNeoXForCausalLM
from nntile.model.gpt_neox_config import GPTNeoXConfig

# Create argument parser
parser = argparse.ArgumentParser(prog="GPTNeoX-based neural networks",
        description="This example presents an NNTile implementation of a "
        "GPTNeoX-family of models and compares it against the Huggingface. "
        "It checks relative accuracy of a forward pass (values of "
        "activations) and backward pass (gradients of parameters and "
        "activations) and a throughput of inference and training. It can "
        "also fine-tune a pretrained NNTile model on a chosen dataset.")

parser.add_argument("--remote_model_name", type=str,
                    default="EleutherAI/gpt-neox-20b")

parser.add_argument("--pretrained", choices=["local", "remote"],
                    default="local")
parser.add_argument("--checkpoint-path", type=str, default="")
parser.add_argument("--config-path", type=str, default="")
parser.add_argument("--save-checkpoint-path", type=str, default=".model")
parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"],
                    default="adam")


parser.add_argument("--model-path", default=".model")
parser.add_argument("--seq-len", type=int, default=512)
parser.add_argument("--seq-len-tile", type=int, default=-1)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--minibatch-size", type=int, default=-1)
parser.add_argument("--minibatch-size-tile", type=int, default=-1)

parser.add_argument("--hidden-size-tile", type=int, default=-1)
parser.add_argument("--intermediate-size-tile", type=int, default=-1)
parser.add_argument("--n-head-tile", type=int, default=-1)

parser.add_argument(
    "--dtype", choices=["fp32", "fp64", "tf32",
                               "bf16", "fp32_fast_fp16",
                               "fp32_fast_bf16"], default="fp32")
parser.add_argument("--restrict", choices=["cpu", "cuda", None],
        default=None)
parser.add_argument("--use-redux", action="store_true")


parser.add_argument("--dataset-path", default=".data")
parser.add_argument("--dataset-file", default="")

parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--nepochs", type=int, default=1)

parser.add_argument("--logger", action="store_true")
parser.add_argument("--logger-server-addr", type=str,
                    default="localhost")
parser.add_argument("--logger-server-port", type=int, default=5001)

# Parse arguments
args = parser.parse_args()
print(args)

if args.seq_len_tile == -1:
    args.seq_len_tile = args.seq_len
if args.minibatch_size == -1:
    args.minibatch_size = args.batch_size
if args.minibatch_size_tile == -1:
    args.minibatch_size_tile = args.minibatch_size
# Check arguments
assert args.seq_len_tile > 0
assert args.batch_size > 0
assert args.minibatch_size > 0
assert args.minibatch_size_tile > 0
assert args.batch_size % args.minibatch_size == 0
num_minibatch = args.batch_size // args.minibatch_size
assert args.minibatch_size % args.minibatch_size_tile == 0
assert args.nepochs > 0

# Load named pretrained PyTorch model
if args.pretrained == "remote":
    # Newer versions of transformers can use fast attention, so we disable it
    # through a parameter attn_implementation
    model_torch = ModelTorch.from_pretrained(args.remote_model_name,
                cache_dir=args.model_path, local_files_only=False)
elif args.pretrained == "local":
    if args.config_path:
        f = open(args.config_path)
        conf_dict = json.load(f)
        f.close()
        config = ConfigTorch(**conf_dict)
        model_torch = ModelTorch(config)
        tokenizer = None
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
            model_torch.load_state_dict(checkpoint['model_state_dict'])

model_torch.eval()
print(model_torch.config)

# Initialize NNTile and StarPU
time0 = time.time()
# Set up StarPU+MPI and init codelets
verbose = 0
context = nntile.Context(
    ncpu=-1,
    ncuda=-1,
    ooc=0,
    logger=args.logger,
    logger_addr=args.logger_server_addr,
    logger_port=args.logger_server_port,
    verbose=0
)
nntile.starpu.profiling_init()
nntile.starpu.profiling_disable()
# Restrict computations to CUDA if possible
if args.restrict == "cuda":
    context.restrict_cuda()
elif args.restrict == "cpu":
    context.restrict_cpu()
time1 = time.time() - time0
print("StarPU + NNTile + MPI init in {} seconds".format(time1))

time0 = time.time()
if args.n_head_tile == -1:
    args.n_head_tile = model_torch.config.num_attention_heads
if args.hidden_size_tile == -1:
    args.hidden_size_tile = model_torch.config.hidden_size
if args.intermediate_size_tile == -1:
    args.intermediate_size_tile = model_torch.config.intermediate_size

config_nntile = GPTNeoXConfig(
    vocab_size=model_torch.config.vocab_size,
    vocab_embed_dim_tile=model_torch.config.hidden_size,
    hidden_size=model_torch.config.hidden_size,
    hidden_size_tile=args.hidden_size_tile,
    intermediate_size=model_torch.config.intermediate_size,
    intermediate_size_tile=args.intermediate_size_tile,
    num_heads=model_torch.config.num_attention_heads,
    num_heads_tile=args.n_head_tile,
    dtype=args.dtype,
    layer_norm_epsilon=model_torch.config.layer_norm_eps,
    max_position_embeddings=model_torch.config.max_position_embeddings,
    num_hidden_layers=model_torch.config.num_hidden_layers,
    redux=args.use_redux,
    bos_token_id=model_torch.config.bos_token_id,
    eos_token_id=model_torch.config.eos_token_id,
    rotary_emb_base=model_torch.config.rotary_emb_base,
)

print(config_nntile)
single_batch_pos_ids = np.arange(
    args.seq_len, dtype=np.int64
).reshape(1, args.seq_len)
pos_ids = np.repeat(single_batch_pos_ids, args.minibatch_size, axis=0)

mask = np.array(
            np.triu(np.ones((args.seq_len, args.seq_len))),
            dtype=bool, order="F"
        )

gpt_neox_nntile = GPTNeoXForCausalLM.from_torch(model_torch,
                                                args.minibatch_size,
                                                args.minibatch_size_tile,
                                                args.seq_len,
                                                args.seq_len_tile,
                                                pos_ids,
                                                mask,
                                                config_nntile)
time1 = time.time() - time0
print("Converting PyTorch model to NNTile",
        "requires {} seconds".format(time1))
del model_torch

# Get train tokens
splitted_datasetfile = args.dataset_file.split("/")
if splitted_datasetfile[-1] == "train.bin":
    train_data = np.memmap(Path(args.dataset_path) /
                            args.dataset_file,
                            dtype=np.uint16, mode='r')
    train_tokens_raw = np.array(train_data, order='F', dtype=np.int64)
    del train_data
else:
    raise ValueError("Only train.bin file is accepted"
                        "for training dataset!")

num_train_tokens = train_tokens_raw.shape[0]

num_train_seq = num_train_tokens // (args.seq_len + 1)
num_train_batches = num_train_seq // args.batch_size
num_train_tokens_truncated = num_train_batches * (args.batch_size
        * (args.seq_len + 1))
train_tokens_trunc = np.array(
    train_tokens_raw[:num_train_tokens_truncated],
    order='F', dtype=np.int64)
train_tokens = train_tokens_trunc.reshape(num_train_batches,
                                    num_minibatch,
                                    args.minibatch_size,
                                    args.seq_len + 1)

time0 = time.time()
batch_input = []
batch_output = []
x_traits = nntile.tensor.TensorTraits(
        [args.seq_len, args.minibatch_size],
        [args.seq_len_tile, args.minibatch_size_tile])
x_distr = [0] * x_traits.grid.nelems
for i in range(num_train_batches):
    minibatch_input = []
    minibatch_output = []
    for j in range(num_minibatch):
        x = nntile.tensor.Tensor_int64(x_traits, x_distr)
        x.from_array(np.asfortranarray(train_tokens[i, j, :, :-1].T))
        minibatch_input.append(x)
        y = nntile.tensor.Tensor_int64(x_traits, x_distr)
        y.from_array(np.asfortranarray(train_tokens[i, j, :, 1:].T))
        minibatch_output.append(y)
    batch_input.append(minibatch_input)
    batch_output.append(minibatch_output)
time1 = time.time() - time0
print("From PyTorch loader to NNTile batches in {} seconds".format(time1))
# Set up learning rate and optimizer for training
if args.optimizer == "adam":
    optimizer = nntile.optimizer.Adam(gpt_neox_nntile.get_parameters(),
            args.lr)
elif args.optimizer == "adamw":
    optimizer = nntile.optimizer.AdamW(gpt_neox_nntile.get_parameters(),
            args.lr)
elif args.optimizer == "sgd":
    optimizer = nntile.optimizer.SGD(gpt_neox_nntile.get_parameters(),
            args.lr)

# Define Cross Entropy loss function
loss = nntile.loss.CrossEntropy.generate_simple(
        gpt_neox_nntile.activations[-1],
        scale=1.0 / (args.batch_size * args.seq_len))
# Set up training pipeline
pipeline = nntile.pipeline.Pipeline(batch_input, batch_output,
        gpt_neox_nntile, optimizer, loss, args.nepochs)
# Print pipeline memory info
pipeline.print_meminfo()
# Warmup training
# nntile.starpu.pause()
nntile.starpu.profiling_enable()
pipeline.train_async()
# nntile.starpu.resume()
nntile.starpu.wait_for_all()
nntile.starpu.profiling_disable()
time1 = time.time() - time0
print("NNTile training time: {} seconds".format(time1))
print("NNTile training throughput tokens/sec: {}".format(
        args.nepochs * num_train_batches * args.batch_size
        * args.seq_len / time1))
nflops_fwd_minibatch = gpt_neox_nntile.get_flops_forward()
nflops_bwd_minibatch = gpt_neox_nntile.get_flops_backward()
nflops_minibatch = nflops_fwd_minibatch + nflops_bwd_minibatch
print("NNTile performance (model flops): {} Tflops/s".format(nflops_minibatch
        * args.nepochs * num_train_batches * num_minibatch
        / time1 * 1e-12))
loss_np = np.zeros((1), dtype=np.float32)
loss.val.to_array(loss_np)
print("NNTile loss on the last batch: {}".format(loss_np[0]))

# Convert back to PyTorch and save checkpoint
model_torch = gpt_neox_nntile.to_torch()
torch.save(
    {
        "model_state_dict": model_torch.state_dict(),
    },
    args.save_checkpoint_path,
)
del model_torch

loss.unregister()
optimizer.unregister()
for batch in batch_input + batch_output:
    for x in batch:
        x.unregister()
gpt_neox_nntile.unregister()
