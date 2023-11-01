# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/gpt2_perf_workflow.py
# GPT2 performance analysis workflow
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @author Aleksandr Katrutsa
# @date 2023-11-01

# Imports
import torch
import nntile
import math
import numpy as np
import time
import sys
from torch import Tensor
import torch.nn as nn
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Model, \
        GPT2Config
from torch.optim import Adam
from torch.optim import SGD
from datasets import load_dataset
from nntile.model.gpt2 import GPT2Config as GPT2Config_nntile, \
        GPT2Model as GPT2Model_nntile
from nntile.tensor import copy_async
from nntile.loss import Frob
import pdb 
from typing import Union, Optional, Tuple, List
from packaging import version
import copy
import argparse
import json

# Create argument parser
parser = argparse.ArgumentParser(prog="GPT2-based neural networks", \
        description="This example presents an NNTile implementation of a " \
        "GPT2-family of models and compares it against the Huggingface. " \
        "It checks relative accuracy of a forward pass (values of " \
        "activations) and backward pass (gradients of parameters and " \
        "activations) and a throughput of inference and training. It can " \
        "also fine-tune a pretrained NNTile model on a chosen dataset.")

parser.add_argument("--mode", choices=["init_local", "init_remote", "train", "test"])
parser.add_argument("--checkpoint_path", type=str, default="")
parser.add_argument("--config_path", type=str, default="")
parser.add_argument("--save_checkpoint_path", type=str, default=".model")
parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")

parser.add_argument("--model-path", default=".model")
parser.add_argument("--seq-len-tile", type=int, default=1024)

parser.add_argument("--num-samples", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--minibatch-size", type=int, default=1)
parser.add_argument("--minibatch-size-tile", type=int, default=1)
parser.add_argument("--n-embd-tile", type=int, default=384)
parser.add_argument("--n-inner-tile", type=int, default=1536)
parser.add_argument("--n-head-tile", type=int, default=-1)
parser.add_argument("--torch-device", choices=["cpu", "cuda", "cuda:0", \
        "cuda:1", "cuda:2", "cuda:3", "cuda:4"], default="cpu")
parser.add_argument("--dtype", choices=["fp32", "fp64"], default="fp32")      
# parser.add_argument("--torch-dtype", choices=["fp32", "fp64"], default="fp32")
# parser.add_argument("--torch-compile", action="store_true")
# parser.add_argument("--nntile-dtype", choices=["fp32", "fp64"], default="fp32")
# parser.add_argument("--torch-nforward", type=int, default=0)
# parser.add_argument("--torch-nforward-warmup", type=int, default=0)
# parser.add_argument("--torch-nbackward", type=int, default=0)
# parser.add_argument("--torch-nbackward-warmup", type=int, default=0)
# parser.add_argument("--nntile-restrict", choices=["cpu", "cuda", None], \
#         default=None)
parser.add_argument("--nntile-flashattention", action="store_true")
parser.add_argument("--nntile-use-redux", action="store_true")
# parser.add_argument("--nntile-nforward", type=int, default=0)
# parser.add_argument("--nntile-nforward-warmup", type=int, default=0)
# parser.add_argument("--nntile-nbackward", type=int, default=0)
# parser.add_argument("--nntile-nbackward-warmup", type=int, default=0)
# parser.add_argument("--dataset", default="WikiText-103")
# parser.add_argument("--dataset-path", default=".data")
# parser.add_argument("--dataset-select", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.0)
parser.add_argument("--nepochs", type=int, default=0)
# parser.add_argument("--torch-nepochs", type=int, default=0)
# parser.add_argument("--torch-nepochs-warmup", type=int, default=0)
# parser.add_argument("--nntile-nepochs", type=int, default=0)
# parser.add_argument("--nntile-nepochs-warmup", type=int, default=0)

args = parser.parse_args()
print(args)

assert args.num_samples % args.batch_size == 0
assert args.batch_size % args.minibatch_size == 0
num_minibatch = args.batch_size // args.minibatch_size
assert args.minibatch_size % args.minibatch_size_tile == 0

# Init model locally or remote and save the corresponding checkpoint for further processing

if args.mode == "init_remote":
    model_torch = GPT2LMHeadModel.from_pretrained("gpt2", \
        cache_dir=args.model_path)
    model_torch.lm_head.weight = nn.Parameter(model_torch.lm_head \
                                    .weight.detach().clone())
    torch.save({'model_state_dict': model_torch.state_dict()}, args.save_checkpoint_path + "/init_checkpoint_remote.pt")
if args.mode == "init_local":
    f = open(args.config_path)
    conf_dict = json.load(f)
    f.close()
    config = GPT2Config(**conf_dict)
    model_torch = GPT2LMHeadModel(config)
    model_torch.lm_head.weight = nn.Parameter(model_torch.lm_head \
        .weight.detach().clone())
    torch.save({'model_state_dict': model_torch.state_dict()}, args.save_checkpoint_path + "/init_checkpoint_local.pt")

if args.mode == "train":
    f = open(args.config_path)
    conf_dict = json.load(f)
    f.close()
    config = GPT2Config(**conf_dict)
    if args.n_head_tile == -1:
        args.n_head_tile = config.n_head
    assert config.n_head % args.n_head_tile == 0
    assert config.n_positions % args.seq_len_tile == 0
    config.attn_pdrop = 0
    config.embd_pdrop = 0
    config.resid_pdrop = 0
    inner_dim = config.n_inner if config.n_inner is not None \
        else 4 * config.hidden_size
    config.n_inner = inner_dim
    model_torch = GPT2LMHeadModel(config).to(args.torch_device)
    model_torch.lm_head.weight = nn.Parameter(model_torch.lm_head \
        .weight.detach().clone())
    if args.optimizer == "adam":
        optimizer = Adam(model_torch.parameters(), args.lr)
    elif args.optimizer == "sgd":
        optimizer = SGD(model_torch.parameters(), args.lr)
    checkpoint = torch.load(args.checkpoint_path)
    model_torch.load_state_dict(checkpoint['model_state_dict'])
    # Forward FLOPs of matrix products per input sequence per GPT block
    nflops_seq_block = 2*config.n_positions*config.n_embd*(3+1)*config.n_embd \
            + 4*config.n_positions*config.n_positions*config.n_embd \
            + 4*config.n_positions*config.n_embd*config.n_inner
    # Forward FLOPs of matrix products per input sequence per GPT model
    nflops_seq = config.num_hidden_layers*nflops_seq_block \
            + 2*config.n_positions*config.n_embd*config.vocab_size

    # Create Nntile copy of the loaded model
    # Initialize NNTile and StarPU
    time0 = time.time()
    # Set up StarPU+MPI and init codelets
    nntile_config = nntile.starpu.Config(-1, -1, 1)
    nntile.starpu.profiling_init()
    nntile.starpu.profiling_disable()
    nntile.starpu.init()
    # Restrict computations to CUDA if possible
    # if args.nntile_restrict == "cuda":
    #     nntile.starpu.restrict_cuda()
    # elif args.nntile_restrict == "cpu":
    #     nntile.starpu.restrict_cpu()
    time1 = time.time() - time0
    print("StarPU + NNTile + MPI init in {} seconds".format(time1))
    next_tag = 0
    nntile_model_config = GPT2Config_nntile(config.vocab_size, args.n_embd_tile, \
        config.n_embd, args.n_embd_tile, config.max_position_embeddings, \
        config.n_inner, args.n_inner_tile, config.layer_norm_epsilon, \
        config.num_hidden_layers, config.n_head, args.n_head_tile, \
        "gelutanh", args.nntile_flashattention, args.nntile_use_redux)
    nntile_model, next_tag = GPT2Model_nntile.from_torch(model_torch, \
            args.minibatch_size, args.minibatch_size_tile, config.n_positions, \
            args.seq_len_tile, nntile_model_config, next_tag)
    # Create random dataset for train sumulation
    input_value = torch.randint(config.vocab_size, \
            (args.num_samples, config.n_positions), dtype=torch.int64, \
            device=args.torch_device)
    input_label = torch.randint(config.vocab_size, \
            (args.num_samples, config.n_positions, ), dtype=torch.int64, \
            device=args.torch_device)
    # Run train loop for n_epoch for PyTorch model and report loss after every epoch
    num_train_batches = args.num_samples // args.batch_size
    loss_func = nn.CrossEntropyLoss(reduction="sum")
    for i in range(1, args.nepochs+1):
        print("Epoch", i)
        for j in range(num_train_batches):
            optimizer.zero_grad()
            train_output = model_torch(input_value[j*args.batch_size:(j+1)*args.batch_size, :])
            train_logits = train_output.logits.reshape(-1, config.vocab_size)
            loss_val = loss_func(train_logits, 
                        input_label[j*args.batch_size:(j+1)*args.batch_size, :].reshape(train_logits.shape[0], ))
            loss_val.backward()
            optimizer.step()
        print("loss in the last batch =", loss_val.item())
    if args.torch_device.startswith("cuda"):
        torch.cuda.synchronize()
    time1 = time.time() - time0
    print("Torch training time: {} seconds".format(time1))
    print("Torch training throughput tokens/sec: {}".format( \
            args.nepochs * num_train_batches * args.batch_size \
            * config.n_positions/time1))
    print("Torch performance: {} Tflops/s".format(3 * nflops_seq \
            * args.nepochs * num_train_batches * args.batch_size \
            / time1 * 1e-12))



    
    



