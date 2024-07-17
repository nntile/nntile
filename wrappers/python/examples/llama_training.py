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

import numpy as np
import torch
# import torch.nn as nn
# from datasets import load_dataset
# from torch.optim import SGD, Adam, AdamW
from transformers import LlamaConfig, LlamaModel

import nntile
# from nntile.layer.llama_mlp import LlamaMLP as LlamaMLP_nntile
# from nntile.loss import Frob
from nntile.model.llama import Llama as Llama_nntile, LlamaConfigNNTile

# from transformers import LlamaTokenizerFast


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


llama_config_torch = LlamaConfig()
print(llama_config_torch)
llama_config_torch.num_hidden_layers = 1
torch_model = LlamaModel(llama_config_torch)
print(torch_model)

llama_config_nntile = LlamaConfigNNTile(
    vocab_size=llama_config_torch.vocab_size,
    vocab_embed_dim_tile=llama_config_torch.hidden_size,
    hidden_size=llama_config_torch.hidden_size,
    hidden_size_tile=llama_config_torch.hidden_size,
    max_position_embeddings=llama_config_torch.max_position_embeddings,
    num_hidden_layers=llama_config_torch.num_hidden_layers,
    rms_norm_eps=llama_config_torch.rms_norm_eps,
    n_attention_head=llama_config_torch.num_attention_heads)

llama_nntile, next_tag = Llama_nntile.from_torch(torch_model,
                                                 10, 10, 4096, 4096,
                                                 llama_config_nntile,
                                                 next_tag)

# from datasets import load_dataset
# import numpy as np
# train_dataset = load_dataset("wikitext", "wikitext-103-v1", \
#                 split='train', cache_dir=".data") \
#                 .select(np.arange(args.dataset_select, dtype=np.int64))
# test_dataset = load_dataset("wikitext", "wikitext-103-v1", \
#                 split='test', cache_dir=".data")

# tokenizer = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b")
# map_train_tokens = map(lambda x: tokenizer(x["text"])["input_ids"],
# train_dataset)
# list_train_tokens = []
# for seq in map_train_tokens:
#     list_train_tokens.extend(seq)
# num_train_tokens = len(list_train_tokens)

n_positions = 4096
# num_train_seq = num_train_tokens // (n_positions+1)
# num_train_batches = num_train_seq // args.batch_size
# num_train_tokens_truncated = num_train_batches * args.batch_size \
#         * (n_positions+1)
# train_tokens = np.array(list_train_tokens[:num_train_tokens_truncated], \
#         order='F', dtype=np.int64)
# train_tokens = train_tokens.reshape(num_train_batches, \
#         num_minibatch, args.minibatch_size, n_positions+1)

input_value = torch.randint(llama_config_torch.vocab_size,
            (10, n_positions), dtype=torch.int64, device=torch_device)
position_ids = torch.zeros((10, n_positions)).to(torch.long).to(torch_device)

torch_model.to(torch_device)
output = torch_model(input_value, position_ids=position_ids)
print(output.last_hidden_state.shape)

# Transfer input/output to NNTile format
llama_nntile.activations[0].value.from_array(input_value.T.cpu().numpy())
llama_nntile.forward_async()
nntile.starpu.wait_for_all()
llama_nntile_output_np = np.zeros((4096, 4096, 10), order="F",
                                  dtype=np.float32)
llama_nntile.activations[-1].value.to_array(llama_nntile_output_np)

output_torch = output.last_hidden_state.cpu().detach().numpy()
print(output_torch.shape, llama_nntile_output_np.shape,
      llama_nntile_output_np.T.shape)
print(np.linalg.norm(output_torch - llama_nntile_output_np.T) /
      np.linalg.norm(output_torch))

llama_nntile.unregister()
