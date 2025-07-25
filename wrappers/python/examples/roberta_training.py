# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/roberta_training.py
# Roberta training example. The original Roberta model is similar
# to the Bert model, but the masking procedure is different
#
# @version 1.1.0

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import SGD, Adam, AdamW
from transformers import RobertaConfig, RobertaForMaskedLM

import nntile
from nntile.model.bert_config import BertConfigNNTile as RobertaConfigNNTile
from nntile.model.roberta import (
    RobertaForMaskedLM as RobertaForMaskedLM_nntile)

# Create argument parser
parser = argparse.ArgumentParser(prog="Roberta neural network",
        description="This example presents an NNTile implementation of a "
        "RoBERTa model and compare them against the Huggingface. "
        "It checks relative accuracy of a forward pass (values of "
        "activations) and backward pass (gradients of parameters and "
        "activations) and a throughput of inference and training. It can "
        "also fine-tune a pretrained NNTile model on a chosen dataset.")

parser.add_argument("--remote_model_name", type=str,
                    default="FacebookAI/roberta-base")

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

parser.add_argument("--dtype", choices=["fp32", "fp64", "fp32_fast_tf32",
                                        "bf16", "fp32_fast_bf16",
                                        "fp32_fast_fp16"],
                    default="fp32")
parser.add_argument("--restrict", choices=["cpu", "cuda", None],
        default=None)
parser.add_argument("--flash-attention", action="store_true")
parser.add_argument("--use-redux", action="store_true")


parser.add_argument("--dataset-path", default=".data")
parser.add_argument("--dataset-file", default="")

parser.add_argument("--lr", type=float, default=0.0)
parser.add_argument("--nepochs", type=int, default=1)
parser.add_argument("--label-mask-token", type=int, default=3)
parser.add_argument("--n-masked-tokens-per-seq", type=int, default=3)

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
    model_torch = RobertaForMaskedLM.from_pretrained(args.remote_model_name,
                cache_dir=args.model_path)
elif args.pretrained == "local":
    if args.config_path:
        f = open(args.config_path)
        conf_dict = json.load(f)
        f.close()
        config = RobertaConfig(**conf_dict)
        model_torch = RobertaForMaskedLM(config)
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
context = nntile.Context(
    ncpu=-1,
    ncuda=-1,
    ooc=0,
    logger=args.logger,
    logger_addr=args.logger_server_addr,
    logger_port=args.logger_server_port,
    verbose=0,
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

roberta_config_nntile = RobertaConfigNNTile(
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
    type_vocab_size=1,
    redux=args.use_redux,
    flashattention=args.flash_attention
)

print(roberta_config_nntile)

roberta_nntile = RobertaForMaskedLM_nntile.from_torch(model_torch,
                                                args.minibatch_size,
                                                args.minibatch_size_tile,
                                                args.seq_len,
                                                args.seq_len_tile,
                                                roberta_config_nntile)
time1 = time.time() - time0
print("Converting PyTorch model to NNTile",
        "requires {} seconds".format(time1))
del model_torch

splitted_datasetfile = args.dataset_file.split("/")
if splitted_datasetfile[-1] == "train.bin":
    train_data = np.memmap(Path(args.dataset_path) /
                            args.dataset_file,
                            dtype=np.uint16, mode='r')
    train_tokens_raw = np.array(train_data, order='F', dtype=np.int64)
    del train_data
else:
    raise ValueError("Only train.bin file is accepted"
                        " for training dataset!")

num_train_tokens = train_tokens_raw.shape[0]

num_train_seq = num_train_tokens // args.seq_len
num_train_batches = num_train_seq // args.batch_size
num_train_tokens_truncated = num_train_batches * (args.batch_size
        * args.seq_len)
train_tokens_trunc = np.array(
    train_tokens_raw[:num_train_tokens_truncated],
    order='F', dtype=np.int64)
train_tokens = train_tokens_trunc.reshape(num_train_batches,
                                    num_minibatch,
                                    args.minibatch_size,
                                    args.seq_len)

time0 = time.time()
batch_masked_data = []
batch_labels = []
x_traits = nntile.tensor.TensorTraits(
        [args.seq_len, args.minibatch_size],
        [args.seq_len_tile, args.minibatch_size_tile])

rng = np.random.default_rng()
for epoch_idx in range(args.nepochs):
    for i in range(num_train_batches):
        minibatch_masked_data = []
        minibatch_labels = []
        for j in range(num_minibatch):
            current_mask = np.zeros((args.minibatch_size,
                                     args.seq_len), dtype=bool)
            idx_masked_tokens = np.array([
                rng.choice(args.seq_len, size=(args.n_masked_tokens_per_seq,),
                           replace=False) for k in range(args.minibatch_size)])
            for k in range(args.minibatch_size):
                current_mask[k, idx_masked_tokens[k]] = 1

            x = nntile.tensor.Tensor_int64(x_traits)
            current_minibatch = train_tokens[i, j, :, :].copy()
            current_minibatch[current_mask] = args.label_mask_token
            x.from_array(np.asfortranarray(current_minibatch).T)
            minibatch_masked_data.append(x)
            y = nntile.tensor.Tensor_int64(x_traits)
            current_label = train_tokens[i, j, :, :].copy()
            inverse_current_mask = np.array(1 - current_mask, dtype=bool)
            # Ignore index = -100
            current_label[inverse_current_mask] = -100
            y.from_array(np.asfortranarray(current_label.T))
            minibatch_labels.append(y)
        batch_masked_data.append(minibatch_masked_data)
        batch_labels.append(minibatch_labels)
time1 = time.time() - time0
print("From PyTorch loader to NNTile batches in {} seconds".format(time1))
# Set up learning rate and optimizer for training
if args.optimizer == "adam":
    optimizer = nntile.optimizer.Adam(roberta_nntile.get_parameters(),
            args.lr)
elif args.optimizer == "adamw":
    optimizer = nntile.optimizer.AdamW(roberta_nntile.get_parameters(),
            args.lr)
elif args.optimizer == "sgd":
    optimizer = nntile.optimizer.SGD(roberta_nntile.get_parameters(),
            args.lr)
# Define Cross Entropy loss function
loss = nntile.loss.CrossEntropy.generate_simple(
        roberta_nntile.activations[-1],
        scale=1.0 / (args.batch_size *
                     args.n_masked_tokens_per_seq))
# Set up training pipeline
pipeline = nntile.pipeline.Pipeline(batch_masked_data, batch_labels,
        roberta_nntile, optimizer, loss, 1)

# Print pipeline memory info
pipeline.print_meminfo()
# Warmup training
# nntile.starpu.pause()
nntile.starpu.profiling_enable()
# Call separate pipeline for the single epoch for every masked data
pipeline.train_async()
# nntile.starpu.resume()
nntile.starpu.wait_for_all()
nntile.starpu.profiling_disable()
time1 = time.time() - time0
print("NNTile training time: {} seconds".format(time1))
print("NNTile training throughput tokens/sec: {}".format(
        args.nepochs * num_train_batches * args.batch_size
        * args.seq_len / time1))
loss_np = np.zeros((1), dtype=np.float32)
loss.val.to_array(loss_np)
print("NNTile loss on the last batch: {}".format(loss_np[0]))
if args.save_checkpoint_path:
    trained_torch_model = roberta_nntile.to_torch()
    torch.save({
                'model_state_dict': trained_torch_model.state_dict(),
                }, args.save_checkpoint_path)
    del trained_torch_model
