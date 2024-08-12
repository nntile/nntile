# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/gpt2_perf_workflow.py
# GPT2 performance analysis workflow
#
# @version 1.1.0

import argparse
import json
import time

import numpy as np
# Imports
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from transformers import GPT2Config, GPT2LMHeadModel

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

parser.add_argument(
    "--mode", choices=["init_local", "init_remote", "train", "test"]
)
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
parser.add_argument(
    "--torch-device",
    choices=["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4"],
    default="cpu",
)
parser.add_argument("--dtype", choices=["fp32", "fp64"], default="fp32")
parser.add_argument(
    "--nntile-restrict", choices=["cpu", "cuda", None], default=None
)
parser.add_argument("--nntile-flashattention", action="store_true")
parser.add_argument("--nntile-use-redux", action="store_true")
parser.add_argument("--lr", type=float, default=0.0)
parser.add_argument("--nepochs", type=int, default=0)

args = parser.parse_args()
print(args)

assert args.num_samples % args.batch_size == 0
assert args.batch_size % args.minibatch_size == 0
num_minibatch = args.batch_size // args.minibatch_size
assert args.minibatch_size % args.minibatch_size_tile == 0

# Init model locally or remote and save the corresponding checkpoint for
# further processing
if args.mode == "init_remote":
    model_torch = GPT2LMHeadModel.from_pretrained(
        "gpt2", cache_dir=args.model_path
    )
    model_torch.lm_head.weight = nn.Parameter(
        model_torch.lm_head.weight.detach().clone()
    )
    torch.save(
        {"model_state_dict": model_torch.state_dict()},
        args.save_checkpoint_path + "/init_checkpoint_remote.pt",
    )
if args.mode == "init_local":
    f = open(args.config_path)
    conf_dict = json.load(f)
    f.close()
    config = GPT2Config(**conf_dict)
    model_torch = GPT2LMHeadModel(config)
    model_torch.lm_head.weight = nn.Parameter(
        model_torch.lm_head.weight.detach().clone()
    )
    torch.save(
        {"model_state_dict": model_torch.state_dict()},
        args.save_checkpoint_path + "/init_checkpoint_local.pt",
    )

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
    inner_dim = (
        config.n_inner
        if config.n_inner is not None
        else 4 * config.hidden_size
    )
    config.n_inner = inner_dim
    model_torch = GPT2LMHeadModel(config).to(args.torch_device)
    model_torch.lm_head.weight = nn.Parameter(
        model_torch.lm_head.weight.detach().clone()
    )
    if args.optimizer == "adam":
        optimizer = Adam(model_torch.parameters(), args.lr)
    elif args.optimizer == "sgd":
        optimizer = SGD(model_torch.parameters(), args.lr)
    checkpoint = torch.load(args.checkpoint_path)
    model_torch.load_state_dict(checkpoint["model_state_dict"])
    # Forward FLOPs of matrix products per input sequence per GPT block
    nflops_seq_block = (
        2 * config.n_positions * config.n_embd * (3 + 1) * config.n_embd
        + 4 * config.n_positions * config.n_positions * config.n_embd
        + 4 * config.n_positions * config.n_embd * config.n_inner
    )
    # Forward FLOPs of matrix products per input sequence per GPT model
    nflops_seq = (
        config.num_hidden_layers * nflops_seq_block
        + 2 * config.n_positions * config.n_embd * config.vocab_size
    )

    # Create Nntile copy of the loaded model
    # Initialize NNTile and StarPU
    time0 = time.time()
    # Set up StarPU+MPI and init codelets
    nntile_config = nntile.starpu.Config(-1, -1, 1)
    nntile.starpu.profiling_init()
    nntile.starpu.profiling_disable()
    nntile.starpu.init()
    # Restrict computations to CUDA if possible
    if args.nntile_restrict == "cuda":
        nntile.starpu.restrict_cuda()
    # elif args.nntile_restrict == "cpu":
    #     nntile.starpu.restrict_cpu()
    time1 = time.time() - time0
    print("StarPU + NNTile + MPI init in {} seconds".format(time1))
    next_tag = 0
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
    # Create random dataset for train sumulation
    num_train_batches = args.num_samples // args.batch_size
    num_minibatch = args.batch_size // args.minibatch_size
    torch.manual_seed(0)
    random_dataset = torch.randint(
        config.vocab_size,
        (
            num_train_batches,
            num_minibatch,
            args.minibatch_size,
            config.n_positions + 1,
        ),
        dtype=torch.int64,
        device=args.torch_device,
    )
    torch_input = random_dataset[:, :, :, :-1]
    torch_output = random_dataset[:, :, :, 1:]
    # Run train loop for n_epoch for PyTorch model and report loss after every
    # epoch
    loss_func = nn.CrossEntropyLoss(reduction="sum")
    torch_loss_hist = []
    for i in range(args.nepochs):
        # print("Epoch={}".format(i))
        for j in range(num_train_batches):
            optimizer.zero_grad()
            loss = torch.zeros(
                1, dtype=torch.float32, device=args.torch_device
            )
            for k in range(num_minibatch):
                train_input = torch_input[j][k].to(args.torch_device)
                train_labels = (
                    torch_output[j][k].to(args.torch_device).reshape(-1)
                )
                train_output = model_torch(train_input)
                train_logits = train_output.logits.reshape(
                    -1, config.vocab_size
                )
                loss_local = loss_func(train_logits, train_labels)
                loss += loss_local
            torch_loss_hist.append(loss.item())
            loss.backward(retain_graph=True)
            # print("loss for batch {}={}".format(j, loss.item()))
            optimizer.step()
    if args.torch_device.startswith("cuda"):
        torch.cuda.synchronize()
    time1 = time.time() - time0
    print("Torch training time: {} seconds".format(time1))
    print(
        "Torch training throughput tokens/sec: {}".format(
            args.nepochs
            * num_train_batches
            * args.batch_size
            * config.n_positions
            / time1
        )
    )
    print(
        "Torch performance: {} Tflops/s".format(
            3
            * nflops_seq
            * args.nepochs
            * num_train_batches
            * args.batch_size
            / time1
            * 1e-12
        )
    )

    # Train with NNtile model
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
            x.from_array(
                np.asfortranarray(random_dataset[i, j, :, :-1].cpu().T)
            )
            minibatch_input.append(x)
            y = nntile.tensor.Tensor_int64(x_traits, x_distr, next_tag)
            next_tag = y.next_tag
            y.from_array(
                np.asfortranarray(random_dataset[i, j, :, 1:].cpu().T)
            )
            minibatch_output.append(y)
        batch_input.append(minibatch_input)
        batch_output.append(minibatch_output)
    time1 = time.time() - time0
    print("From PyTorch loader to NNTile batches in {} seconds".format(time1))
    # Set up learning rate and optimizer for training
    if args.optimizer == "sgd":
        nntile_optimizer = nntile.optimizer.SGD(
            nntile_model.get_parameters(), args.lr, next_tag
        )
    elif args.optimizer == "adam":
        nntile_optimizer = nntile.optimizer.FusedAdam(
            nntile_model.get_parameters(), args.lr, next_tag
        )
    next_tag = nntile_optimizer.get_next_tag()
    # Define Cross Entropy loss function
    loss, next_tag = nntile.loss.CrossEntropy.generate_simple(
        nntile_model.activations[-1], next_tag
    )
    # Set up training pipeline
    pipeline = nntile.pipeline.Pipeline(
        batch_input,
        batch_output,
        nntile_model,
        nntile_optimizer,
        loss,
        args.nepochs,
    )
    # Warmup training
    pipeline.train_async()
    nntile.starpu.wait_for_all()
    time0 = time.time()
    nntile.starpu.profiling_disable()
    time1 = time.time() - time0
    for i in range(len(torch_loss_hist)):
        assert (
            abs(torch_loss_hist[i] - pipeline.loss_hist[i])
            / torch_loss_hist[i]
            < 1e-5
        )
    print("NNTile training time: {} seconds".format(time1))
    print(
        "NNTile training throughput tokens/sec: {}".format(
            args.nepochs
            * num_train_batches
            * args.batch_size
            * config.n_positions
            / time1
        )
    )
    print(
        "NNTile performance: {} Tflops/s".format(
            3
            * nflops_seq
            * args.nepochs
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
    nntile_optimizer.unregister()
    for batch in batch_input + batch_output:
        for x in batch:
            x.unregister()

# Unregister all tensors related to model
nntile_model.unregister()
