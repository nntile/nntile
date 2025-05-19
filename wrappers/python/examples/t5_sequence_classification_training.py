# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/t5_sequence_classification_training.py
# T5ForSequenceClassification training example
#
# @version 1.1.0
# ruff: noqa: E501

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import SGD, Adam, AdamW
from transformers import T5Config as T5ConfigTorch
from transformers.models.t5.modeling_t5 import (
    T5ForSequenceClassification as T5ForSequenceClassification_torch)

import nntile
from nntile.model.t5_config import T5ConfigNNTile
from nntile.model.t5_model import T5ForSequenceClassification
from nntile.tensor import TensorMoments

# Create argument parser
parser = argparse.ArgumentParser(
    prog="T5-based sequence classification",
    description="This example presents an NNTile implementation of a "
    "T5 model for sequence classification tasks and compares it against "
    "the Huggingface. It checks relative accuracy of a forward pass (values of "
    "activations) and backward pass (gradients of parameters and "
    "activations) and a throughput of inference and training. It can "
    "also fine-tune a pretrained NNTile model on a chosen dataset.",
)

parser.add_argument("--remote_model_name", type=str, default="google/flan-t5-small")

parser.add_argument("--pretrained", choices=["local", "remote"], default="remote")
parser.add_argument("--checkpoint-path", type=str, default="")
parser.add_argument("--config-path", type=str, default="")
parser.add_argument("--save-checkpoint-path", type=str, default=".model")
parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="adam")

parser.add_argument("--model-path", default=".model")
parser.add_argument("--seq-len", type=int, default=512)
parser.add_argument("--seq-len-tile", type=int, default=-1)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--minibatch-size", type=int, default=-1)
parser.add_argument("--minibatch-size-tile", type=int, default=-1)

parser.add_argument("--d-model-tile", type=int, default=-1)
parser.add_argument("--d-ff-tile", type=int, default=-1)
parser.add_argument("--num-heads-tile", type=int, default=-1)
parser.add_argument("--num-labels", type=int, default=2)

parser.add_argument(
    "--dtype",
    choices=["fp32", "fp64", "tf32", "bf16", "fp32_fast_fp16", "fp32_fast_bf16"],
    default="fp32",
)
parser.add_argument("--restrict", choices=["cpu", "cuda", None], default=None)
parser.add_argument("--use-redux", action="store_true")

parser.add_argument("--dataset-path", default=".data")
parser.add_argument("--dataset-file", default="")

parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--nepochs", type=int, default=1)

parser.add_argument("--logger", action="store_true")
parser.add_argument("--logger-server-addr", type=str, default="localhost")
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
    model_torch = T5ForSequenceClassification_torch.from_pretrained(
        args.remote_model_name, cache_dir=args.model_path, local_files_only=False
    )
elif args.pretrained == "local":
    if args.config_path:
        f = open(args.config_path)
        conf_dict = json.load(f)
        f.close()
        config = T5ConfigTorch(**conf_dict)
        model_torch = T5ForSequenceClassification_torch(config)
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
            model_torch.load_state_dict(checkpoint["model_state_dict"])

model_torch.eval()
print(model_torch.config)

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
print("StarPU + NNTile + MPI init in {} seconds".format(time1))
next_tag = 0

time0 = time.time()
if args.num_heads_tile == -1:
    args.num_heads_tile = model_torch.config.num_heads
if args.d_model_tile == -1:
    args.d_model_tile = model_torch.config.d_model
if args.d_ff_tile == -1:
    args.d_ff_tile = model_torch.config.d_ff

t5_config_nntile = T5ConfigNNTile(
    vocab_size=model_torch.config.vocab_size,
    d_model=model_torch.config.d_model,
    d_model_tile=args.d_model_tile,
    d_ff=model_torch.config.d_ff,
    d_ff_tile=args.d_ff_tile,
    d_kv=model_torch.config.d_kv,
    d_kv_tile=model_torch.config.d_kv,  # d_kv_tile=args.d_kv_tile,
    num_layers=model_torch.config.num_layers,
    # num_decoder_layers=model_torch.config.num_decoder_layers,
    n_head=model_torch.config.num_heads,
    n_head_tile=args.num_heads_tile,
    # relative_attention_num_buckets=model_torch.config.relative_attention_num_buckets,
    dropout_rate=0.0,  # model_torch.config.dropout_rate,
    layer_norm_epsilon=model_torch.config.layer_norm_epsilon,
    # initializer_factor=model_torch.config.initializer_factor,
    # feed_forward_proj=model_torch.config.feed_forward_proj,
    # dtype=args.dtype,
    redux=args.use_redux,
)

print(t5_config_nntile)

# Create input tensor for T5 model
x_traits = nntile.tensor.TensorTraits(
    [args.seq_len, args.minibatch_size], [args.seq_len_tile, args.minibatch_size_tile]
)
x_distr = [0] * x_traits.grid.nelems
x = nntile.tensor.Tensor_int64(x_traits, x_distr, next_tag)
x_grad = nntile.tensor.Tensor_int64(x_traits, x_distr, next_tag)
next_tag = x.next_tag

# Create dummy array with input tokens for initialization
x_np = np.ones((args.seq_len, args.minibatch_size), dtype=np.int64, order="F")
x.from_array(x_np)
x_tm = TensorMoments(x, None, False)

x_decoder = nntile.tensor.Tensor_int64(x_traits, x_distr, next_tag)
x_decoder_grad = nntile.tensor.Tensor_int64(x_traits, x_distr, next_tag)
next_tag = x_decoder.next_tag

# x_tm_decoder = TensorMoments(x_decoder, None, False)

# Convert PyTorch model to NNTile
t5_model, next_tag = T5ForSequenceClassification.from_torch(
    model_torch, x_tm, x_tm, t5_config_nntile, next_tag
)
time1 = time.time() - time0
print("Converting PyTorch model to NNTile", "requires {} seconds".format(time1))
del model_torch

# Load training dataset - assuming dataset format with input_ids and labels
# Note: This is a simplified dataset loading assuming preprocessed data
# In a real application, you would use a proper dataset loader
splitted_datasetfile = args.dataset_file.split("/")
if splitted_datasetfile[-1].endswith(".npz"):
    train_data = np.load(Path(args.dataset_path) / args.dataset_file)
    train_input_ids = train_data["input_ids"]
    train_labels = train_data["labels"]
else:
    # For demonstration purposes, create dummy data
    print("Using dummy training data")
    num_train_samples = 1000
    vocab_size = 32000
    data_ids = np.random.randint(
        0, 32000, size=(num_train_samples, args.seq_len + 1), dtype=np.int64
    )
    train_input_ids = data_ids[:, :-1]
    train_labels = data_ids[:, 1:]

    # train_input_ids = np.ones((num_train_samples, args.seq_len), dtype=np.int64)
    # train_labels = np.random.randint(0, args.num_labels, size=(num_train_samples,), dtype=np.int64)

num_train_samples = train_input_ids.shape[0]
num_train_batches = num_train_samples // args.batch_size

# Prepare NNTile tensors for input and output
time0 = time.time()
batch_input = []
batch_output = []
x_traits = nntile.tensor.TensorTraits(
    [args.seq_len, args.minibatch_size], [args.seq_len_tile, args.minibatch_size_tile]
)
x_distr = [0] * x_traits.grid.nelems

y_traits = nntile.tensor.TensorTraits(
    [1, args.minibatch_size], [1, args.minibatch_size_tile]
)
y_distr = [0] * y_traits.grid.nelems

for i in range(num_train_batches):
    minibatch_input = []
    minibatch_output = []
    for j in range(num_minibatch):
        # Calculate index range for current minibatch
        start_idx = i * args.batch_size + j * args.minibatch_size
        end_idx = start_idx + args.minibatch_size

        # Create input tensor
        x = nntile.tensor.Tensor_int64(x_traits, x_distr, next_tag)
        next_tag = x.next_tag
        x.from_array(np.asfortranarray(train_input_ids[start_idx:end_idx, :].T))
        print("add minibatch x: ", x.shape)
        minibatch_input.append(x)

        # Create output tensor (labels)
        y = nntile.tensor.Tensor_int64(x_traits, x_distr, next_tag)
        next_tag = y.next_tag
        y.from_array(np.asfortranarray(train_labels[start_idx:end_idx, :].T))
        print("add minibatch y: ", y.shape)
        minibatch_output.append(y)

    batch_input.append(minibatch_input)
    batch_output.append(minibatch_output)
time1 = time.time() - time0
print("From PyTorch loader to NNTile batches in {} seconds".format(time1))

# Set up learning rate and optimizer for training
if args.optimizer == "adam":
    optimizer = nntile.optimizer.Adam(t5_model.get_parameters(), args.lr, next_tag)
elif args.optimizer == "adamw":
    optimizer = nntile.optimizer.AdamW(t5_model.get_parameters(), args.lr, next_tag)
elif args.optimizer == "sgd":
    optimizer = nntile.optimizer.SGD(t5_model.get_parameters(), args.lr, next_tag)
next_tag = optimizer.get_next_tag()

# Define Cross Entropy loss function for classification
loss, next_tag = nntile.loss.CrossEntropy.generate_simple(
    t5_model.activations[-1], next_tag, scale=1.0 / args.batch_size
)

# Set up training pipeline
pipeline = nntile.pipeline.Pipeline(
    batch_input, batch_output, t5_model, optimizer, loss, args.nepochs
)

# Print pipeline memory info
pipeline.print_meminfo()

# Start training
time0 = time.time()
nntile.starpu.profiling_enable()
pipeline.train_async()
nntile.starpu.wait_for_all()
nntile.starpu.profiling_disable()
time1 = time.time() - time0
print("NNTile training time: {} seconds".format(time1))
print(
    "NNTile training throughput samples/sec: {}".format(
        args.nepochs * num_train_batches * args.batch_size / time1
    )
)

# Calculate and report performance metrics
nflops_fwd_minibatch = t5_model.get_flops_forward()
nflops_bwd_minibatch = t5_model.get_flops_backward()
nflops_minibatch = nflops_fwd_minibatch + nflops_bwd_minibatch
print(
    "NNTile performance (model flops): {} Tflops/s".format(
        nflops_minibatch
        * args.nepochs
        * num_train_batches
        * num_minibatch
        / time1
        * 1e-12
    )
)

# Report final loss
loss_np = np.zeros((1), dtype=np.float32)
loss.val.to_array(loss_np)
print("NNTile loss on the last batch: {}".format(loss_np[0]))

# Convert back to PyTorch and save checkpoint
model_torch = t5_model.to_torch()
torch.save(
    {
        "model_state_dict": model_torch.state_dict(),
    },
    args.save_checkpoint_path,
)
del model_torch

# Clean up resources
loss.unregister()
optimizer.unregister()
for batch in batch_input + batch_output:
    for x in batch:
        x.unregister()
t5_model.unregister()
