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
import json
import time

import numpy as np
import torch
import torch.nn as nn
# from datasets import load_dataset
from torch.optim import SGD, Adam, AdamW
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast

import nntile
# from nntile.layer.llama_mlp import LlamaMLP as LlamaMLP_nntile
# from nntile.loss import Frob
from nntile.model.llama_causal import LlamaForCausalLM as Llama_nntile
from nntile.model.llama_config import LlamaConfigNNTile

# from transformers import LlamaTokenizerFast


# Create argument parser
parser = argparse.ArgumentParser(prog="LLaMa-based neural networks",
        description="This example presents an NNTile implementation of a "
        "LLaMa-family of models and compares it against the Huggingface. "
        "It checks relative accuracy of a forward pass (values of "
        "activations) and backward pass (gradients of parameters and "
        "activations) and a throughput of inference and training. It can "
        "also fine-tune a pretrained NNTile model on a chosen dataset.")

# parser.add_argument("--remote_model_name", default="huggyllama/llama-7b")
parser.add_argument("--remote_model_name", default="kimihailv/llama-1.3b")

parser.add_argument("--pretrained", choices=["local", "remote"],
                    default="local")
parser.add_argument("--checkpoint_path", type=str, default="")
parser.add_argument("--config_path", type=str, default="")
parser.add_argument("--save_checkpoint_path", type=str, default=".model")
parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"],
                    default="adam")


parser.add_argument("--model-path", default=".model")
parser.add_argument("--seq-len", type=int, default=1024)
parser.add_argument("--seq-len-tile", type=int, default=1024)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--minibatch-size", type=int, default=1)
parser.add_argument("--minibatch-size-tile", type=int, default=1)
parser.add_argument("--hidden-size-tile", type=int)
parser.add_argument("--intermediate-size-tile", type=int)
parser.add_argument("--n-head-tile", type=int)
parser.add_argument("--torch-device", choices=["cpu", "cuda", "cuda:0",
        "cuda:1", "cuda:2", "cuda:3", "cuda:4"], default="cpu")
parser.add_argument("--torch-dtype", choices=["fp32", "fp64", "bf16"],
                    default="fp32")
# parser.add_argument("--torch-compile", action="store_true")
parser.add_argument("--nntile-dtype", choices=["fp32", "fp64", "tf32", "bf16"],
                    default="fp32")

# parser.add_argument("--torch-nforward", type=int, default=0)
# parser.add_argument("--torch-nforward-warmup", type=int, default=0)
# parser.add_argument("--torch-nbackward", type=int, default=0)
# parser.add_argument("--torch-nbackward-warmup", type=int, default=0)
parser.add_argument("--nntile-restrict", choices=["cpu", "cuda", None],
        default=None)
# parser.add_argument("--nntile-flashattention", action="store_true")
# parser.add_argument("--nntile-use-redux", action="store_true")
# parser.add_argument("--nntile-nforward", type=int, default=0)
# parser.add_argument("--nntile-nforward-warmup", type=int, default=0)
# parser.add_argument("--nntile-nbackward", type=int, default=0)
# parser.add_argument("--nntile-nbackward-warmup", type=int, default=0)
parser.add_argument("--dataset", default="WikiText-103")
parser.add_argument("--dataset-path", default=".data")
parser.add_argument("--dataset-select", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.0)
parser.add_argument("--torch-nepochs", type=int, default=0)
# parser.add_argument("--torch-nepochs-warmup", type=int, default=0)
parser.add_argument("--nntile-nepochs", type=int, default=0)
# parser.add_argument("--nntile-nepochs-warmup", type=int, default=0)
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
# assert args.n_embd_tile > 0
# assert args.n_inner_tile > 0
# assert args.torch_nforward >= 0
# assert args.torch_nbackward >= 0
# assert args.torch_nepochs >= 0
# assert args.nntile_nforward >= 0
# assert args.nntile_nbackward >= 0
# assert args.nntile_nepochs >= 0

# Set Torch default device to cpu
# torch.set_default_device("cpu")
torch_device = args.torch_device

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
    model_torch = LlamaForCausalLM.from_pretrained(args.remote_model_name,
                cache_dir=args.model_path, local_files_only=True)
elif args.pretrained == "local":
    if args.config_path:
        f = open(args.config_path)
        conf_dict = json.load(f)
        f.close()
        config = LlamaConfig(**conf_dict)
        model_torch = LlamaForCausalLM(config).to(torch_dtype)
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

tokenizer = LlamaTokenizerFast.from_pretrained(args.remote_model_name,
                                                cache_dir=args.model_path)
model_torch.eval()
model_torch.to(torch_device).to(torch_dtype)
print(model_torch.config)

if args.nntile_nepochs:
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

    time0 = time.time()
    llama_config_nntile = LlamaConfigNNTile(
        vocab_size=model_torch.vocab_size,
        vocab_embed_dim_tile=model_torch.config.hidden_size,
        hidden_size=model_torch.config.hidden_size,
        hidden_size_tile=args.hidden_size_tile,
        max_position_embeddings=model_torch.config.max_position_embeddings,
        num_hidden_layers=model_torch.config.num_hidden_layers,
        rms_norm_eps=model_torch.config.rms_norm_eps,
        n_attention_head=model_torch.config.num_attention_heads,
        num_key_value_heads=model_torch.config.num_key_value_heads,
        intermediate_size=model_torch.config.intermediate_size,
        intermediate_size_tile=args.intermediate_size_tile,
        n_head_tile=args.n_head_tile,
        dtype=args.nntile_dtype
        )
    print(llama_config_nntile)

    single_batch_pos_ids = np.arange(args.seq_len).reshape(1, args.seq_len)
    pos_ids = np.repeat(single_batch_pos_ids, args.minibatch_size, axis=0)

    mask = np.array(np.triu(np.ones((args.seq_len, args.seq_len))),
                        dtype=bool, order="F")
    llama_nntile, next_tag = Llama_nntile.from_torch(model_torch,
                                                    args.minibatch_size,
                                                    args.minibatch_size_tile,
                                                    args.seq_len,
                                                    args.seq_len_tile,
                                                    pos_ids,
                                                    mask,
                                                    llama_config_nntile,
                                                    next_tag)
    time1 = time.time() - time0
    print("Converting PyTorch model to NNTile",
          "requires {} seconds".format(time1))
if args.torch_nepochs == 0:
    del model_torch

if args.torch_nepochs > 0 or args.nntile_nepochs > 0:
    import numpy as np
    from datasets import load_dataset
    train_dataset = load_dataset("wikitext", "wikitext-103-v1",
                    split='train', cache_dir=".data").select(np.arange(
                        args.dataset_select, dtype=np.int64))
    test_dataset = load_dataset("wikitext", "wikitext-103-v1",
                    split='test', cache_dir=".data")

    map_train_tokens = map(lambda x: tokenizer(x["text"])["input_ids"],
                        train_dataset)
    list_train_tokens = []
    for seq in map_train_tokens:
        list_train_tokens.extend(seq)
    num_train_tokens = len(list_train_tokens)

    num_train_seq = num_train_tokens // (args.seq_len + 1)
    num_train_batches = num_train_seq // args.batch_size
    num_train_tokens_truncated = num_train_batches * (args.batch_size
            * (args.seq_len + 1))
    train_tokens = np.array(list_train_tokens[:num_train_tokens_truncated],
            order='F', dtype=np.int64)
    train_tokens = train_tokens.reshape(num_train_batches,
            num_minibatch, args.minibatch_size, args.seq_len + 1)

if args.torch_nepochs > 0:
    torch_input = []
    torch_output = []
    for i in range(num_train_batches):
        minibatch_input = []
        minibatch_output = []
        for j in range(num_minibatch):
            minibatch_input.append(torch.tensor(train_tokens[i, j, :, :-1],
                requires_grad=False).contiguous())
            minibatch_output.append(torch.tensor(train_tokens[i, j, :, 1:],
                requires_grad=False).contiguous())
        torch_input.append(minibatch_input)
        torch_output.append(minibatch_output)
    loss_func = nn.CrossEntropyLoss(reduction="mean")
    model_torch.train()
    if args.pretrained == "remote":
        if args.optimizer == "adam":
            optimizer = Adam(model_torch.parameters(), args.lr)
        elif args.optimizer == "sgd":
            optimizer = SGD(model_torch.parameters(), args.lr)
        elif args.optimizer == "adamw":
            optimizer = AdamW(model_torch.parameters(), args.lr)
        else:
            raise ValueError
    # Warmup training
    # for i in range(args.torch_nepochs_warmup):
    #     for j in range(num_train_batches):
    #         optimizer.zero_grad()
    #         loss = torch.zeros(1, dtype=torch_dtype,
    #                            device=args.torch_device)
    #         for k in range(num_minibatch):
    #             train_input = torch_input[j][k].to(args.torch_device)
    #             train_labels = torch_output[j][k].to(
    #                               args.torch_device).reshape(-1)
    #             train_output = model_torch.to(torch_dtype)(train_input)
    #             train_logits = train_output.logits.reshape(-1,
    #                                                        config.vocab_size)
    #             loss_local = loss_func(train_logits, train_labels)
    #             loss_local.backward()
    #             loss += loss_local
    #         print("loss={}".format(loss.item()), flush=True)
    #         optimizer.step()
    # Actual training
    if args.torch_device.startswith("cuda"):
        torch.cuda.synchronize()
    time0 = time.time()
    for i in range(args.torch_nepochs):
        for j in range(num_train_batches):
            optimizer.zero_grad()
            loss = torch.zeros(1, dtype=torch_dtype, device=args.torch_device)
            for k in range(num_minibatch):
                train_input = torch_input[j][k].to(
                    args.torch_device)
                train_labels = torch_output[j][k].to(
                    args.torch_device).reshape(-1)
                train_output = model_torch(train_input)
                train_logits = train_output.logits.reshape(-1,
                                                           model_torch.config.vocab_size)
                loss_local = loss_func(train_logits, train_labels)
                loss_local.backward()
                loss += loss_local
            print("Batch={}/{} Epoch={}/{} Loss={}".format(j + 1,
                                                           num_train_batches,
                                                           i + 1,
                                                           args.torch_nepochs,
                                                           loss.item()),
                                                           flush=True)
            optimizer.step()
    if args.torch_device.startswith("cuda"):
        torch.cuda.synchronize()
    time1 = time.time() - time0
    print("Torch training time: {} seconds".format(time1), flush=True)
    print("Torch training throughput tokens/sec: {}".format(
            args.torch_nepochs * num_train_batches * args.batch_size
            * args.seq_len / time1), flush=True)
    # print("Torch performance: {} Tflops/s".format(3 * torch_nflops_seq \
    #         * args.torch_nepochs * num_train_batches * args.batch_size \
    #         / time1 * 1e-12), flush=True)
    print("Torch loss on the last batch: {}".format(loss.item()), flush=True)

if args.nntile_nepochs > 0:
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
        optimizer = nntile.optimizer.Adam(llama_nntile.get_parameters(),
                args.lr, next_tag)
    elif args.optimizer == "adamw":
        optimizer = nntile.optimizer.AdamW(llama_nntile.get_parameters(),
                args.lr, next_tag)
    elif args.optimizer == "sgd":
        optimizer = nntile.optimizer.SGD(llama_nntile.get_parameters(),
                args.lr, next_tag)
    next_tag = optimizer.get_next_tag()
    # Define Cross Entropy loss function
    loss, next_tag = nntile.loss.CrossEntropy.generate_simple(
            llama_nntile.activations[-1], next_tag,
            scale=1.0 / (args.batch_size * args.seq_len))
    # Set up training pipeline
    pipeline = nntile.pipeline.Pipeline(batch_input, batch_output,
            llama_nntile, optimizer, loss, args.nntile_nepochs)
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
            args.nntile_nepochs * num_train_batches * args.batch_size
            * args.seq_len / time1))
    # print("NNTile performance: {} Tflops/s".format(nflops_seq \
    #         * args.nntile_nepochs * num_train_batches * args.batch_size \
    #         / time1 * 1e-12))
    loss_np = np.zeros((1), dtype=np.float32)
    loss.val.to_array(loss_np)
    print("NNTile loss on the last batch: {}".format(loss_np[0]))
    loss.unregister()
    optimizer.unregister()
    for batch in batch_input + batch_output:
        for x in batch:
            x.unregister()
    llama_nntile.unregister()
