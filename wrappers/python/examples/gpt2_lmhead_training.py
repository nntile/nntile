# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/gpt2_lmhead_training.py
# GPT2LMHead training example
#
# @version 1.1.0

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from transformers import GPT2Config as GPT2ConfigTorch
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2LMHeadModel as GPT2LMHead_torch)

import nntile
from nntile.model.gpt2_config import GPT2ConfigNNTile
from nntile.model.gpt2_lmhead import GPT2LMHead

# Create argument parser
parser = argparse.ArgumentParser(prog="GPT2-based neural networks",
        description="This example presents an NNTile implementation of a "
        "GPT2-family of models and compares it against the Huggingface. "
        "It checks relative accuracy of a forward pass (values of "
        "activations) and backward pass (gradients of parameters and "
        "activations) and a throughput of inference and training. It can "
        "also fine-tune a pretrained NNTile model on a chosen dataset.")

parser.add_argument("--remote_model_name", type=str,
                    default="openai-community/gpt2")

parser.add_argument("--pretrained", choices=["local", "remote"],
                    default="local")
parser.add_argument("--checkpoint-path", type=str, default="")
parser.add_argument("--config-path", type=str, default="")
parser.add_argument("--save-checkpoint-path", type=str, default=".model")
parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"],
                    default="adam")


parser.add_argument("--model-path", default=".model")
parser.add_argument("--seq-len", type=int, default=1024)
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
parser.add_argument("--flash-attention", action="store_true")
parser.add_argument("--use-redux", action="store_true")


parser.add_argument("--dataset-path", default=".data")
parser.add_argument("--dataset-file", default="")

parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--nepochs", type=int, default=1)

parser.add_argument("--logger", action="store_true")
parser.add_argument("--logger-server-addr", type=str,
                    default="localhost")
parser.add_argument("--logger-server-port", type=int, default=5001)

parser.add_argument("--ooc", action="store_true")
parser.add_argument("--ooc-path", type=str, default="/tmp/nntile_ooc")
parser.add_argument("--ooc-size", type=int, default=1073741824)

parser.add_argument("--force-offload-disk-portion-parameters", type=float,
                    default=0.0)
parser.add_argument("--force-offload-disk-portion-gradients", type=float,
                    default=0.0)
parser.add_argument("--force-offload-disk-portion-activations", type=float,
                    default=0.0)
parser.add_argument("--force-offload-disk-portion-temporaries", type=float,
                    default=0.0)
parser.add_argument("--force-offload-disk-portion-optimizer", type=float,
                    default=0.0)

parser.add_argument("--force-offload-ram-portion-parameters", type=float,
                    default=0.0)
parser.add_argument("--force-offload-ram-portion-gradients", type=float,
                    default=0.0)
parser.add_argument("--force-offload-ram-portion-activations", type=float,
                    default=0.0)
parser.add_argument("--force-offload-ram-portion-temporaries", type=float,
                    default=0.0)
parser.add_argument("--force-offload-ram-portion-optimizer", type=float,
                    default=0.0)

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
assert args.force_offload_disk_portion_parameters >= 0.0
assert args.force_offload_disk_portion_parameters <= 1.0
assert args.force_offload_disk_portion_gradients >= 0.0
assert args.force_offload_disk_portion_gradients <= 1.0
assert args.force_offload_disk_portion_activations >= 0.0
assert args.force_offload_disk_portion_activations <= 1.0
assert args.force_offload_disk_portion_temporaries >= 0.0
assert args.force_offload_disk_portion_temporaries <= 1.0
assert args.force_offload_disk_portion_optimizer >= 0.0
assert args.force_offload_disk_portion_optimizer <= 1.0

assert args.force_offload_ram_portion_parameters >= 0.0
assert args.force_offload_ram_portion_parameters <= 1.0
assert args.force_offload_ram_portion_gradients >= 0.0
assert args.force_offload_ram_portion_gradients <= 1.0
assert args.force_offload_ram_portion_activations >= 0.0
assert args.force_offload_ram_portion_activations <= 1.0
assert args.force_offload_ram_portion_temporaries >= 0.0
assert args.force_offload_ram_portion_temporaries <= 1.0
assert args.force_offload_ram_portion_optimizer >= 0.0
assert args.force_offload_ram_portion_optimizer <= 1.0

# Load named pretrained PyTorch model
if args.pretrained == "remote":
    # Newer versions of transformers can use fast attention, so we disable it
    # through a parameter attn_implementation
    model_torch = GPT2LMHead_torch.from_pretrained(args.remote_model_name,
                cache_dir=args.model_path, local_files_only=False)
elif args.pretrained == "local":
    if args.config_path:
        f = open(args.config_path)
        conf_dict = json.load(f)
        f.close()
        config = GPT2ConfigTorch(**conf_dict)
        model_torch = GPT2LMHead_torch(config)
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

model_torch.lm_head.weight = nn.Parameter(
    model_torch.lm_head.weight.detach().clone()
)
inner_dim = (
    model_torch.config.n_inner if model_torch.config.n_inner is not None else
      4 * model_torch.config.hidden_size
)
model_torch.config.n_inner = inner_dim

# Initialize NNTile and StarPU
time0 = time.time()
nntile.nntile_init(
    ncpus=-1,
    ncuda=-1,
    cublas=1,
    ooc=0,
    logger=args.logger,
    logger_server_addr=args.logger_server_addr,
    logger_server_port=args.logger_server_port)
nntile.starpu.profiling_init()
nntile.starpu.profiling_disable()
# Restrict computations to CUDA if possible
if args.restrict == "cuda":
    nntile.starpu.restrict_cuda()
elif args.restrict == "cpu":
    nntile.starpu.restrict_cpu()
time1 = time.time() - time0
print("StarPU + NNTile + MPI init in {} seconds".format(time1))

time0 = time.time()
if args.n_head_tile == -1:
    args.n_head_tile = model_torch.config.n_head
if args.hidden_size_tile == -1:
    args.hidden_size_tile = model_torch.config.n_embd
if args.intermediate_size_tile == -1:
    args.intermediate_size_tile = model_torch.config.n_inner

gpt2_config_nntile = GPT2ConfigNNTile(
    vocab_size=model_torch.config.vocab_size,
    vocab_embed_dim_tile=model_torch.config.n_embd,
    hidden_size=model_torch.config.n_embd,
    hidden_size_tile=args.hidden_size_tile,
    max_position_embeddings=model_torch.config.max_position_embeddings,
    num_hidden_layers=model_torch.config.num_hidden_layers,
    layer_norm_epsilon=model_torch.config.layer_norm_epsilon,
    n_head=model_torch.config.n_head,
    intermediate_size=model_torch.config.n_inner,
    intermediate_size_tile=args.intermediate_size_tile,
    n_head_tile=args.n_head_tile,
    dtype=args.dtype,
    flash_attention=args.flash_attention
)

print(gpt2_config_nntile)

gpt2lmhead_nntile = GPT2LMHead.from_torch(model_torch,
                                                args.minibatch_size,
                                                args.minibatch_size_tile,
                                                args.seq_len,
                                                args.seq_len_tile,
                                                gpt2_config_nntile)
time1 = time.time() - time0
print("Converting PyTorch model to NNTile",
        "requires {} seconds".format(time1))
del model_torch

# Set forced offloading to disk for parameters, gradients and activations
gpt2lmhead_nntile.force_offload_disk_parameters(args.force_offload_disk_portion_parameters)
gpt2lmhead_nntile.force_offload_disk_gradients(args.force_offload_disk_portion_gradients)
gpt2lmhead_nntile.force_offload_disk_activations(args.force_offload_disk_portion_activations)
gpt2lmhead_nntile.force_offload_disk_temporaries(args.force_offload_disk_portion_temporaries)

# Set forced offloading to RAM for parameters, gradients and activations
gpt2lmhead_nntile.force_offload_ram_parameters(args.force_offload_ram_portion_parameters)
gpt2lmhead_nntile.force_offload_ram_gradients(args.force_offload_ram_portion_gradients)
gpt2lmhead_nntile.force_offload_ram_activations(args.force_offload_ram_portion_activations)
gpt2lmhead_nntile.force_offload_ram_temporaries(args.force_offload_ram_portion_temporaries)

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
    optimizer = nntile.optimizer.Adam(gpt2lmhead_nntile.get_parameters(),
            args.lr)
elif args.optimizer == "adamw":
    optimizer = nntile.optimizer.AdamW(gpt2lmhead_nntile.get_parameters(),
            args.lr)
elif args.optimizer == "sgd":
    optimizer = nntile.optimizer.SGD(gpt2lmhead_nntile.get_parameters(),
            args.lr, next_tag)
next_tag = optimizer.get_next_tag()

# Set OOC force for optimizer
optimizer.force_offload_disk(args.force_offload_disk_portion_optimizer)
# Set RAM force for optimizer
optimizer.force_offload_ram(args.force_offload_ram_portion_optimizer)

# Define Cross Entropy loss function
loss = nntile.loss.CrossEntropy.generate_simple(
        gpt2lmhead_nntile.activations[-1],
        scale=1.0 / (args.batch_size * args.seq_len))
# Set up training pipeline
pipeline = nntile.pipeline.Pipeline(batch_input, batch_output,
        gpt2lmhead_nntile, optimizer, loss, args.nepochs)
# Print pipeline memory info
pipeline.print_meminfo()
# nntile.starpu.pause()
nntile.starpu.profiling_init()
nntile.starpu.profiling_enable()
pipeline.train_async()
# nntile.starpu.resume()
nntile.starpu.wait_for_all()
nntile.starpu.profiling_bus_display_summary()
nntile.starpu.profiling_disable()
time1 = time.time() - time0
print("NNTile training time: {} seconds".format(time1))
print("NNTile training throughput tokens/sec: {}".format(
        args.nepochs * num_train_batches * args.batch_size
        * args.seq_len / time1))
nflops_fwd_minibatch = gpt2lmhead_nntile.get_flops_forward()
nflops_bwd_minibatch = gpt2lmhead_nntile.get_flops_backward()
nflops_minibatch = nflops_fwd_minibatch + nflops_bwd_minibatch
print("NNTile performance (model flops): {} Tflops/s".format(nflops_minibatch
        * args.nepochs * num_train_batches * num_minibatch
        / time1 * 1e-12))
loss_np = np.zeros((1), dtype=np.float32)
loss.val.to_array(loss_np)
print("NNTile loss on the last batch: {}".format(loss_np[0]))
model_torch = gpt2lmhead_nntile.to_torch()
torch.save({
            'model_state_dict': model_torch.state_dict(),
            }, args.save_checkpoint_path)
