# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/performance/llama_perf.py
# Llama performance test script
#
# @version 1.1.0

import argparse
import json
import time

import numpy as np
import torch
from transformers import LlamaConfig
# from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, LlamaMLP)

import nntile
# from nntile.model.llama_causal import LlamaForCausalLM as Llama_nntile
from nntile.model.llama_config import LlamaConfigNNTile
from nntile.model.llama_decoder import LlamaDecoder as LlamaDecoder_nntile
from nntile.model.llama_mlp import LlamaMLP as LlamaMLP_nntile
from nntile.tensor import TensorMoments, TensorTraits

# Create argument parser
parser = argparse.ArgumentParser(prog="Test performance script for LLaMa",
        description="This example presents an NNTile implementation of a "
        "LLaMa-family of models and compares it against the Huggingface. "
        "It checks relative accuracy of a forward pass (values of "
        "activations) and backward pass (gradients of parameters and "
        "activations) and a throughput of inference and training. It can "
        "also fine-tune a pretrained NNTile model on a chosen dataset.")

parser.add_argument("--config-path", type=str, default="")
parser.add_argument("--submodule", choices=["mlp", "decoder", "causal_llama"],
                    default="mlp")

parser.add_argument("--seq-len", type=int, default=1024)
parser.add_argument("--seq-len-tile", type=int, default=-1)
parser.add_argument("--minibatch-size", type=int, default=1)
parser.add_argument("--minibatch-size-tile", type=int, default=-1)

parser.add_argument("--hidden-size-tile", type=int, default=-1)
parser.add_argument("--intermediate-size-tile", type=int, default=-1)
parser.add_argument("--n-head-tile", type=int, default=-1)

parser.add_argument("--dtype", choices=["fp32", "fp32_fast_tf32", "bf16",
                                        "fp32_fast_fp16", "fp32_fast_bf16"],
                    default="fp32")
parser.add_argument("--restrict", choices=["cpu", "cuda", None],
                    default=None)
parser.add_argument("--flash-attention", action="store_true")
parser.add_argument("--use-redux", action="store_true")
parser.add_argument("--torch-compile", action="store_true")
parser.add_argument("--num_warmup_calls", type=int, default=1)
parser.add_argument("--logger", action="store_true")
parser.add_argument("--logger-server-addr", type=str,
                    default="localhost")
parser.add_argument("--logger-server-port", type=int, default=5001)

# Parse arguments
args = parser.parse_args()
print(args)

if args.seq_len_tile == -1:
    args.seq_len_tile = args.seq_len
if args.minibatch_size_tile == -1:
    args.minibatch_size_tile = args.minibatch_size
# Check arguments
assert args.seq_len_tile > 0
assert args.minibatch_size > 0
assert args.minibatch_size_tile > 0
assert args.minibatch_size % args.minibatch_size_tile == 0

dtype2nntile = {
        'fp32': nntile.tensor.Tensor_fp32,
        'fp32_fast_tf32': nntile.tensor.Tensor_fp32_fast_tf32,
        'bf16': nntile.tensor.Tensor_bf16,
        'fp32_fast_fp16': nntile.tensor.Tensor_fp32_fast_fp16,
        'fp32_fast_bf16': nntile.tensor.Tensor_fp32_fast_bf16,
}


f = open(args.config_path)
conf_dict = json.load(f)
f.close()
llama_torch_config = LlamaConfig(**conf_dict)

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
    torch_device = "cuda"
elif args.restrict == "cpu":
    nntile.starpu.restrict_cpu()
    torch_device = "cpu"

time1 = time.time() - time0
print("StarPU + NNTile + MPI init in {} seconds".format(time1))
next_tag = 0

time0 = time.time()
if args.n_head_tile == -1:
    args.n_head_tile = llama_torch_config.num_attention_heads
if args.hidden_size_tile == -1:
    args.hidden_size_tile = llama_torch_config.hidden_size
if args.intermediate_size_tile == -1:
    args.intermediate_size_tile = llama_torch_config.intermediate_size

llama_config_nntile = LlamaConfigNNTile(
    vocab_size=llama_torch_config.vocab_size,
    vocab_embed_dim_tile=llama_torch_config.hidden_size,
    hidden_size=llama_torch_config.hidden_size,
    hidden_size_tile=args.hidden_size_tile,
    max_position_embeddings=llama_torch_config.max_position_embeddings,
    num_hidden_layers=llama_torch_config.num_hidden_layers,
    rms_norm_eps=llama_torch_config.rms_norm_eps,
    n_attention_head=llama_torch_config.num_attention_heads,
    num_key_value_heads=llama_torch_config.num_key_value_heads,
    intermediate_size=llama_torch_config.intermediate_size,
    intermediate_size_tile=args.intermediate_size_tile,
    n_head_tile=args.n_head_tile,
    dtype=args.dtype,
    flash_attention=args.flash_attention
)

print(llama_config_nntile)

gen = np.random.default_rng(42)
if args.submodule == "mlp":
    x_shape = [llama_config_nntile.hidden_size,
               args.seq_len, args.minibatch_size]
    input_data = gen.standard_normal(x_shape, dtype=np.float32)
    x_nntile = np.array(input_data, dtype=np.float32, order="F")
    x_type = dtype2nntile[args.dtype]
    x_torch = torch.Tensor(x_nntile.T)
    x_basetile = [args.hidden_size_tile,
                  args.seq_len_tile,
                  args.minibatch_size_tile]
    x_traits = TensorTraits(x_shape, x_basetile)
    x_distr = [0] * x_traits.grid.nelems
    x_value = x_type(x_traits, x_distr, 0)
    x_value.from_array(x_nntile)
    x_grad = x_type(x_traits, x_distr, 0)
    X = TensorMoments(x_value, x_grad, grad_required=True)
    torch_layer_ = LlamaMLP(llama_torch_config)
    time0 = time.time()
    nntile_module, _ = LlamaMLP_nntile.from_torch(torch_layer_, X,
                                                llama_config_nntile, 0)
    time1 = time.time() - time0
    print("Converting PyTorch model to NNTile requires ",
          "{} seconds".format(time1))
elif args.submodule == "decoder":
    x_shape = [llama_config_nntile.hidden_size,
               args.seq_len, args.minibatch_size]
    x_basetile = [args.hidden_size_tile,
                  args.seq_len_tile,
                  args.minibatch_size_tile]
    x_traits = TensorTraits(x_shape, x_basetile)
    x_distr = [0] * x_traits.grid.nelems
    x_type = dtype2nntile[args.dtype]
    x_value = x_type(x_traits, x_distr, 0)
    x_grad = x_type(x_traits, x_distr, 0)
    X = TensorMoments(x_value, x_grad, grad_required=True)
    gen = np.random.default_rng(42)
    x_random = gen.standard_normal(x_shape, dtype=np.float32)
    x_nntile = np.array(x_random, dtype=np.float32, order="F")
    x_value.from_array(x_nntile)
    x_torch = torch.tensor(x_nntile.T, requires_grad=True)
    pos_ids = gen.integers(args.seq_len,
                           size=(args.minibatch_size, args.seq_len),
                           dtype=np.int64)
    mask = np.array(np.triu(np.ones((args.seq_len, args.seq_len))),
                    dtype=bool, order="F")
    # TODO: Move this option to args
    llama_torch_config._attn_implementation = "eager"
    # layer_idx input param is None since
    # we do not use caching and explcitly state this
    torch_layer_ = LlamaDecoderLayer(llama_torch_config,
                                     layer_idx=None).to(torch_device)
    mask_torch = torch.Tensor(np.array(1 - mask, dtype=np.float32)).T \
            * torch.finfo(torch.float32).min
    mask_torch = mask_torch[None, None, :, :].expand(args.minibatch_size,
                                            1, -1, -1).to(torch_device)
    time0 = time.time()
    nntile_module, _ = LlamaDecoder_nntile.from_torch(torch_layer_, X,
                                                     pos_ids, mask,
                                                     llama_config_nntile,
                                                     0)
    time1 = time.time() - time0
    print("Converting PyTorch model to NNTile requires ",
          "{} seconds".format(time1))
elif args.submodule == "causal_llama":
    raise ValueError("Causal LLaMa isnot supported yet!")

if args.torch_compile:
    torch_layer = torch.compile(torch_layer_)
    torch.set_float32_matmul_precision('high')
else:
    torch_layer = torch_layer_


torch_layer = torch_layer.to(torch_device)
torch_layer.eval()
x_torch = x_torch.to(torch_device)

for n_wup in range(args.num_warmup_calls):
    if args.submodule == "mlp":
        output = torch_layer(x_torch)
    elif args.submodule == "decoder":
        output = torch_layer(x_torch,
                             position_ids=torch.tensor(pos_ids).to(torch_device),
                             attention_mask=mask_torch)

start_torch_time = time.time()
if args.submodule == "mlp":
    output = torch_layer(x_torch)
elif args.submodule == "decoder":
    output = torch_layer(x_torch,
                         position_ids=torch.tensor(pos_ids).to(torch_device),
                         attention_mask=mask_torch)
fin_torch_time = time.time()

for n_wup in range(args.num_warmup_calls):
    nntile_module.forward_async()
    nntile.starpu.wait_for_all()

start_nntile_time = time.time()
nntile.starpu.profiling_enable()
nntile_module.forward_async()
nntile.starpu.wait_for_all()
nntile.starpu.profiling_disable()
fin_nntile_time = time.time()

nntile_module.unregister()
print("NNTile timing = {}".format(fin_nntile_time - start_nntile_time))
print("PyTorch timing = {}".format(fin_torch_time - start_torch_time))
