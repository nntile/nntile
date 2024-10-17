# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/llama_training.py
# Llama training example
#
# @version 1.1.0

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import SGD, Adam, AdamW
from transformers import BertConfig, BertForMaskedLM

import nntile
from nntile.model.bert import BertForMaskedLM as BertForMaskedLM_nntile
from nntile.model.bert_config import BertConfigNNTile

# Create argument parser
parser = argparse.ArgumentParser(prog="Bert neural networks",
        description="This example presents an NNTile implementation of a "
        "Bert/RoBERTa models and compare them against the Huggingface. "
        "It checks relative accuracy of a forward pass (values of "
        "activations) and backward pass (gradients of parameters and "
        "activations) and a throughput of inference and training. It can "
        "also fine-tune a pretrained NNTile model on a chosen dataset.")

parser.add_argument("--remote_model_name", type=str,
                    default="bert-base-uncased")

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

parser.add_argument("--dtype", choices=["fp32", "fp64", "tf32", "bf16"],
                    default="fp32")
parser.add_argument("--restrict", choices=["cpu", "cuda", None],
        default=None)
parser.add_argument("--flash-attention", action="store_true")
parser.add_argument("--use-redux", action="store_true")


parser.add_argument("--dataset-path", default=".data")
parser.add_argument("--dataset-file", default="")

parser.add_argument("--lr", type=float, default=0.0)
parser.add_argument("--nepochs", type=int, default=1)
parser.add_argument("--n_masks_per_seq", type=int, default=1)
parser.add_argument("--label_mask_token", type=int, default=3)
parser.add_argument("--n_masked_tokens_per_seq", type=int, default=3)

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
    model_torch = BertForMaskedLM.from_pretrained(args.remote_model_name,
                cache_dir=args.model_path, local_files_only=True)
elif args.pretrained == "local":
    if args.config_path:
        f = open(args.config_path)
        conf_dict = json.load(f)
        f.close()
        config = BertConfig(**conf_dict)
        model_torch = BertForMaskedLM(config)
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
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model_torch.eval()
print(model_torch.config)

# Initialize NNTile and StarPU
time0 = time.time()
# Set up StarPU+MPI and init codelets
nntile_config = nntile.starpu.Config(-1, -1, 1, args.logger,
        args.logger_server_addr, args.logger_server_port)
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
if args.n_head_tile == -1:
    args.n_head_tile = model_torch.config.num_attention_heads
if args.hidden_size_tile == -1:
    args.hidden_size_tile = model_torch.config.hidden_size
if args.intermediate_size_tile == -1:
    args.intermediate_size_tile = model_torch.config.intermediate_size

bert_config_nntile = BertConfigNNTile(
    vocab_size=model_torch.config.vocab_size,
    vocab_embed_dim_tile=model_torch.config.hidden_size,
    hidden_size=model_torch.config.hidden_size,
    hidden_size_tile=args.hidden_size_tile,
    max_position_embeddings=model_torch.config.max_position_embeddings,
    num_hidden_layers=model_torch.config.num_hidden_layers,
    num_attention_heads=model_torch.config.num_attention_heads,
    intermediate_size=model_torch.config.intermediate_size,
    intermediate_size_tile=args.intermediate_size_tile,
    n_head_tile=args.n_head_tile,
    dtype=args.dtype,
    # flash_attention=args.flash_attention
)

print(bert_config_nntile)

bert_nntile, next_tag = BertForMaskedLM_nntile.from_torch(model_torch,
                                                args.minibatch_size,
                                                args.minibatch_size_tile,
                                                args.seq_len,
                                                args.seq_len_tile,
                                                bert_config_nntile,
                                                next_tag)
time1 = time.time() - time0
print("Converting PyTorch model to NNTile",
        "requires {} seconds".format(time1))
del model_torch

splitted_darasetfile = args.dataset_file.split("/")
if splitted_darasetfile[-1] == "train.bin":
    train_data = np.memmap(Path(args.dataset_path) /
                            args.dataset_file,
                            dtype=np.uint16, mode='r')
    train_tokens_raw = np.array(train_data, order='F', dtype=np.int64)
    del train_data
else:
    raise ValueError("Only train.bin file is accepted"
                        " for training dataset!")

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
batch_masked_data = []
batch_labels = []
x_traits = nntile.tensor.TensorTraits(
        [args.seq_len, args.minibatch_size],
        [args.seq_len_tile, args.minibatch_size_tile])
x_distr = [0] * x_traits.grid.nelems

rng = np.random.default_rng()
for mask_idx in range(args.n_masks_per_seq):
    for i in range(num_train_batches):
        minibatch_masked_data = []
        minibatch_labels = []
        for j in range(num_minibatch):
            current_mask = np.zeros((args.minibatch_size,
                                     args.seq_len), dtype=bool)
            idx_masked_tokens = rng.choice(args.seq_len,
                                           size=(args.minibatch_size,
                                                 args.n_masked_tokens_per_seq))
            current_mask[:, idx_masked_tokens] = 1
            x = nntile.tensor.Tensor_int64(x_traits, x_distr, next_tag)
            next_tag = x.next_tag
            current_minibatch = train_tokens[i, j, :, :-1].copy()
            current_minibatch[current_mask] = args.label_mask_token
            x.from_array(np.asfortranarray(current_minibatch).T)
            minibatch_masked_data.append(x)
            y = nntile.tensor.Tensor_int64(x_traits, x_distr, next_tag)
            next_tag = y.next_tag
            y.from_array(np.asfortranarray(train_tokens[i, j, :, :-1].T))
            minibatch_labels.append(y)
        batch_masked_data.append(minibatch_masked_data)
        batch_labels.append(minibatch_labels)
time1 = time.time() - time0
print("From PyTorch loader to NNTile batches in {} seconds".format(time1))
# Set up learning rate and optimizer for training
if args.optimizer == "adam":
    optimizer = nntile.optimizer.Adam(bert_nntile.get_parameters(),
            args.lr, next_tag)
elif args.optimizer == "adamw":
    optimizer = nntile.optimizer.AdamW(bert_nntile.get_parameters(),
            args.lr, next_tag)
elif args.optimizer == "sgd":
    optimizer = nntile.optimizer.SGD(bert_nntile.get_parameters(),
            args.lr, next_tag)
next_tag = optimizer.get_next_tag()
# Define Cross Entropy loss function
loss, next_tag = nntile.loss.CrossEntropy.generate_simple(
        bert_nntile.activations[-1], next_tag,
        scale=1.0 / (args.batch_size * args.n_masks_per_seq *
                     args.n_masked_tokens_per_seq))
# Set up training pipeline
pipeline = nntile.pipeline.Pipeline(batch_masked_data, batch_labels,
        bert_nntile, optimizer, loss, args.nepochs)
print(batch_masked_data[0][0].shape)
print(bert_nntile.layers[0].w.value.shape)
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
# nflops_fwd_minibatch = bert_nntile.get_flops_forward()
# nflops_bwd_minibatch = bert_nntile.get_flops_backward()
# nflops_minibatch = nflops_fwd_minibatch + nflops_bwd_minibatch
# print("NNTile performance (model flops): {} Tflops/s".format(nflops_minibatch
#         * args.nepochs * num_train_batches * num_minibatch
#         / time1 * 1e-12))
loss_np = np.zeros((1), dtype=np.float32)
loss.val.to_array(loss_np)
print("NNTile loss on the last batch: {}".format(loss_np[0]))
loss.unregister()
optimizer.unregister()
for batch in batch_masked_data + batch_labels:
    for x in batch:
        x.unregister()
bert_nntile.unregister()
