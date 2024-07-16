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
# @version 1.0.0

import argparse
# import json
import time

# import numpy as np
import torch

import nntile

# import torch.nn as nn
# from datasets import load_dataset
# from torch.optim import SGD, Adam, AdamW
# from transformers import LlamaConfig, LlamaModel

# from nntile.layer.llama_mlp import LlamaMLP as LlamaMLP_nntile
# from nntile.loss import Frob
# from nntile.model.llama import LlamaConfig as LlamaConfig_nntile

# Create argument parser
parser = argparse.ArgumentParser(prog="Llama-based neural networks",
        description="This example presents an NNTile implementation of a "
        "Llama-family of models and compares it against the Huggingface. "
        "It checks relative accuracy of a forward pass (values of "
        "activations) and backward pass (gradients of parameters and "
        "activations) and a throughput of inference and training. It can "
        "also fine-tune a pretrained NNTile model on a chosen dataset.")
parser.add_argument("--model", default="llama")

parser.add_argument("--pretrained", choices=["local", "remote"],
                    default="remote")
parser.add_argument("--checkpoint_path", type=str, default="")
parser.add_argument("--config_path", type=str, default="")
parser.add_argument("--save_checkpoint_path", type=str, default=".model")
parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"],
                    default="adam")


parser.add_argument("--model-path", default=".model")
parser.add_argument("--seq-len-tile", type=int, default=1024)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--minibatch-size", type=int, default=1)
parser.add_argument("--minibatch-size-tile", type=int, default=1)
parser.add_argument("--n-embd-tile", type=int, default=384)
parser.add_argument("--n-inner-tile", type=int, default=1536)
parser.add_argument("--n-head-tile", type=int, default=-1)
parser.add_argument("--torch-device", choices=["cpu", "cuda", "cuda:0",
        "cuda:1", "cuda:2", "cuda:3", "cuda:4"], default="cpu")
parser.add_argument("--torch-dtype", choices=["fp32", "fp64", "bf16"],
                    default="fp32")
parser.add_argument("--torch-compile", action="store_true")
parser.add_argument("--nntile-dtype", choices=["fp32", "fp64", "tf32", "bf16"],
                    default="fp32")
parser.add_argument("--check", action="store_true")
parser.add_argument("--check-fp64", action="store_true")
parser.add_argument("--torch-nforward", type=int, default=0)
parser.add_argument("--torch-nforward-warmup", type=int, default=0)
parser.add_argument("--torch-nbackward", type=int, default=0)
parser.add_argument("--torch-nbackward-warmup", type=int, default=0)
parser.add_argument("--nntile-restrict", choices=["cpu", "cuda", None],
        default=None)
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
parser.add_argument("--nntile-logger-server-addr", type=str,
                    default="localhost")
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

# Initialize NNTile and StarPU
time0 = time.time()
# Set up StarPU+MPI and init codelets
nntile_config = nntile.starpu.Config(-1, -1, 1, args.nntile_logger,
        args.nntile_logger_server_addr, args.nntile_logger_server_port)
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
