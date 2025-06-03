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
    LlamaAttention, LlamaDecoderLayer, LlamaMLP)

import nntile
from nntile.layer.llama_attention import (
    LlamaAttention as LlamaAttention_nntile)
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
parser.add_argument("--submodule", choices=["mlp", "decoder",
                                            "attention", "causal_llama"],
                    default="mlp")

parser.add_argument("--attn-implementation",
                    choices=["eager", "sdpa", "flash_attention_2"],
                    default="eager")

parser.add_argument("--use-torch", action="store_true")
parser.add_argument("--use-nntile", action="store_true")

parser.add_argument("--n-fwd", type=int, default=0)
parser.add_argument("--n-fwd-bwd", type=int, default=0)


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
parser.add_argument("--use-redux", action="store_true")
parser.add_argument("--torch-compile", action="store_true")
parser.add_argument("--num-warmup-calls", type=int, default=1)
parser.add_argument("--logger", action="store_true")
parser.add_argument("--logger-server-addr", type=str,
                    default="localhost")
parser.add_argument("--logger-server-port", type=int, default=5001)

# Parse arguments
args = parser.parse_args()
print(args)

if args.use_torch and args.use_nntile:
    raise ValueError("The single run supports only PyTorch run or NNTile")

if not args.use_torch and not args.use_nntile:
    raise ValueError("Please select the backend",
                     "for testing Llama submodules performance")

if args.use_nntile and args.torch_compile:
    raise Warning("--torch-compile flag works only with --use-torch")

if args.seq_len_tile == -1:
    args.seq_len_tile = args.seq_len
if args.minibatch_size_tile == -1:
    args.minibatch_size_tile = args.minibatch_size
# Check arguments
assert args.seq_len_tile > 0
assert args.minibatch_size > 0
assert args.minibatch_size_tile > 0
assert args.minibatch_size % args.minibatch_size_tile == 0
assert (args.n_fwd == 0 and args.n_fwd_bwd > 0) or \
       (args.n_fwd > 0 and args.n_fwd_bwd == 0)

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
llama_torch_config._attn_implementation = args.attn_implementation

if args.use_nntile:
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
    time1 = time.time() - time0
    print("StarPU + NNTile + MPI init in {} seconds".format(time1))
# Restrict computations to CUDA if possible
if args.restrict == "cuda":
    if args.use_nntile:
        nntile.starpu.restrict_cuda()
    if args.use_torch:
        torch_device = "cuda"
elif args.restrict == "cpu":
    if args.use_nntile:
        nntile.starpu.restrict_cpu()
    if args.use_torch:
        torch_device = "cpu"

time0 = time.time()
if args.n_head_tile == -1:
    args.n_head_tile = llama_torch_config.num_attention_heads
if args.hidden_size_tile == -1:
    args.hidden_size_tile = llama_torch_config.hidden_size
if args.intermediate_size_tile == -1:
    args.intermediate_size_tile = llama_torch_config.intermediate_size

if args.use_nntile:
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
    torch_layer_ = LlamaMLP(llama_torch_config)
    x_shape = [llama_torch_config.hidden_size,
                args.seq_len, args.minibatch_size]
    input_data = gen.standard_normal(x_shape, dtype=np.float32)
    x_torch = torch.Tensor(np.array(input_data,
                                    dtype=np.float32,
                                    order="F").T)
    if args.use_nntile:
        x_nntile = np.array(input_data, dtype=np.float32, order="F")
        x_type = dtype2nntile[args.dtype]

        x_basetile = [args.hidden_size_tile,
                    args.seq_len_tile,
                    args.minibatch_size_tile]
        x_traits = TensorTraits(x_shape, x_basetile)
        x_value = x_type(x_traits)
        x_value.from_array(x_nntile)
        x_grad = x_type(x_traits)
        X = TensorMoments(x_value, x_grad, grad_required=True)

        time0 = time.time()
        nntile_module, _ = LlamaMLP_nntile.from_torch(torch_layer_, X,
                                                    llama_config_nntile, 0)
        time1 = time.time() - time0
        print("Converting PyTorch model to NNTile requires ",
            "{} seconds".format(time1))
        del torch_layer_
elif args.submodule == "decoder":
    # layer_idx input param is None since
    # we do not use caching and explcitly state this
    torch_layer_ = LlamaDecoderLayer(llama_torch_config,
                                     layer_idx=None)
    mask = np.array(np.triu(np.ones((args.seq_len, args.seq_len))),
                        dtype=bool, order="F")
    pos_ids = gen.integers(args.seq_len,
                            size=(args.minibatch_size, args.seq_len),
                            dtype=np.int64)
    if args.use_torch:
        mask_torch = torch.Tensor(np.array(1 - mask, dtype=np.float32)).T \
                * torch.finfo(torch.float32).min
        mask_torch = mask_torch[None, None, :, :].expand(args.minibatch_size,
                                                1, -1, -1).to(torch_device)
        pos_ids_torch = torch.tensor(pos_ids).to(torch_device)
    gen = np.random.default_rng(42)
    x_shape = [llama_torch_config.hidden_size,
                args.seq_len, args.minibatch_size]
    x_random = gen.standard_normal(x_shape, dtype=np.float32)
    x_torch = torch.tensor(np.array(x_random, dtype=np.float32, order="F").T,
                           requires_grad=True)
    if args.use_nntile:

        x_basetile = [args.hidden_size_tile,
                    args.seq_len_tile,
                    args.minibatch_size_tile]
        x_traits = TensorTraits(x_shape, x_basetile)
        x_type = dtype2nntile[args.dtype]
        x_value = x_type(x_traits)
        x_grad = x_type(x_traits)
        X = TensorMoments(x_value, x_grad, grad_required=True)
        x_nntile = np.array(x_random, dtype=np.float32, order="F")
        x_value.from_array(x_nntile)

        time0 = time.time()
        nntile_module, _ = LlamaDecoder_nntile.from_torch(torch_layer_, X,
                                                        pos_ids, mask,
                                                        llama_config_nntile,
                                                        0)
        time1 = time.time() - time0
        print("Converting PyTorch model to NNTile requires ",
            "{} seconds".format(time1))
        del torch_layer_
elif args.submodule == "attention":
    # layer_idx input param is None since
    # we do not use caching and explcitly state this
    torch_layer_ = LlamaAttention(
        llama_torch_config, layer_idx=None
    )
    mask = np.array(np.triu(np.ones((args.seq_len, args.seq_len))),
                        dtype=bool, order="F")
    pos_ids = gen.integers(args.seq_len,
                            size=(args.minibatch_size, args.seq_len),
                            dtype=np.int64)
    if args.use_torch:
        mask_torch = torch.Tensor(np.array(1 - mask, dtype=np.float32)).T \
                * torch.finfo(torch.float32).min
        mask_torch = mask_torch[None, None, :, :].expand(args.minibatch_size,
                                                1, -1, -1).to(torch_device)
        pos_ids_torch = torch.tensor(pos_ids).to(torch_device)
    x_shape = [llama_torch_config.hidden_size,
                args.seq_len, args.minibatch_size]
    x_random = gen.standard_normal(x_shape, dtype=np.float32)
    x_torch = torch.tensor(np.array(x_random,
                                    dtype=np.float32,
                                    order="F").T, requires_grad=True)
    if args.use_nntile:
        x_basetile = [args.hidden_size_tile,
                    args.seq_len_tile,
                    args.minibatch_size_tile]
        x_traits = TensorTraits(x_shape, x_basetile)
        x_type = dtype2nntile[args.dtype]
        x_value = x_type(x_traits)
        x_grad = x_type(x_traits)
        X = TensorMoments(x_value, x_grad, grad_required=True)
        x_nntile = np.array(x_random, dtype=np.float32, order="F")
        x_value.from_array(x_nntile)

        pos_ids = gen.integers(args.seq_len,
                            size=(args.minibatch_size, args.seq_len),
                            dtype=np.int64)
        time0 = time.time()
        nntile_module, _ = LlamaAttention_nntile.from_torch(
                torch_layer_, X, pos_ids, mask, llama_config_nntile, 0)
        time1 = time.time() - time0
        print("Converting PyTorch model to NNTile requires ",
            "{} seconds".format(time1))
        del torch_layer_
elif args.submodule == "causal_llama":
    raise ValueError("Causal LLaMa is not supported yet!")

if args.use_torch and args.torch_compile:
    torch_layer = torch.compile(torch_layer_)
elif args.use_torch:
    torch_layer = torch_layer_

if args.n_fwd > 0:
    n_runs = args.n_fwd
elif args.n_fwd_bwd > 0:
    n_runs = args.n_fwd_bwd

if args.use_torch:
    if args.dtype == "bf16":
        torch_layer = torch_layer.bfloat16()
        x_torch = x_torch.bfloat16()
        if args.submodule in ("decoder", "attention"):
            pos_ids_torch = pos_ids_torch.bfloat16()
            mask_torch = mask_torch.bfloat16()
    if args.dtype == "fp32_fast_bf16":
        torch.set_float32_matmul_precision('medium')
    if args.dtype == "fp32_fast_tf32":
        torch.set_float32_matmul_precision('high')
    if args.dtype == "fp32":
        torch.set_float32_matmul_precision('highest')
    torch_layer = torch_layer.to(torch_device)
    torch_layer.eval()
    x_torch = x_torch.to(torch_device)

    for n_wup in range(args.num_warmup_calls):
        if args.submodule == "mlp":
            output = torch_layer(x_torch)
        elif args.submodule == "decoder":
            output = torch_layer(x_torch,
                                position_ids=pos_ids_torch,
                                attention_mask=mask_torch)[0]
        elif args.submodule == "attention":
            output = torch_layer(x_torch,
                            position_ids=pos_ids_torch,
                            attention_mask=mask_torch)[0]
        if args.n_fwd_bwd > 0:
            loss = torch.sum(output)
            loss.backward()

    if torch_device == "cuda":
        torch.cuda.synchronize()

    start_torch_time = time.time()
    for n_fwd_idx in range(n_runs):
        if args.submodule == "mlp":
            output = torch_layer(x_torch)
        elif args.submodule == "decoder":
            output = torch_layer(x_torch,
                            position_ids=pos_ids_torch,
                            attention_mask=mask_torch)[0]
        elif args.submodule == "attention":
            output = torch_layer(x_torch,
                            position_ids=pos_ids_torch,
                            attention_mask=mask_torch)[0]
        if args.n_fwd_bwd > 0:
            loss = torch.sum(output)
            loss.backward()
    if torch_device == "cuda":
        torch.cuda.synchronize()
    fin_torch_time = time.time()
    if args.n_fwd_bwd > 0:
        print("PyTorch timing averaged over {} runs fwd + bwd = {}".format(
                                n_runs,
                                (fin_torch_time - start_torch_time) /
                                n_runs))
    elif args.n_fwd > 0:
        print("PyTorch timing averaged over {} runs of only fwd = {}".format(
                                n_runs,
                                (fin_torch_time - start_torch_time) /
                                n_runs))

if args.use_nntile:

    for n_wup in range(args.num_warmup_calls):
        nntile_module.forward_async()
        if args.n_fwd_bwd > 0:
            nntile_module.clear_gradients()
            if args.submodule in ("mlp", "decoder"):
                nntile_module.activations[-1].grad.from_array(
                    np.ones(nntile_module.activations[-1].value.shape,
                    np.float32, 'F'))
            elif args.submodule == "attention":
                nntile_module.y.grad.from_array(
                    np.ones(nntile_module.y.value.shape,
                    np.float32, 'F'))
            nntile_module.backward_async()
        nntile.starpu.wait_for_all()

    start_nntile_time = time.time()
    nntile.starpu.profiling_enable()
    for run_idx in range(n_runs):
        nntile_module.forward_async()
        if args.n_fwd_bwd > 0:
            nntile_module.clear_gradients()
            if args.submodule in ("mlp", "decoder"):
                nntile_module.activations[-1].grad.from_array(
                    np.ones(nntile_module.activations[-1].value.shape,
                    np.float32, 'F'))
            elif args.submodule == "attention":
                nntile_module.y.grad.from_array(
                    np.ones(nntile_module.y.value.shape,
                    np.float32, 'F'))
            nntile_module.backward_async()
        nntile.starpu.wait_for_all()
    nntile.starpu.profiling_disable()
    fin_nntile_time = time.time()

    nntile_module.unregister()
    if args.submodule == "attention":
        nntile_module.x.unregister()
        nntile_module.y.unregister()
    if args.n_fwd_bwd > 0:
        print("NNTile timing averaged over {} runs of fwd + bwd = {}".format(
                    n_runs,
                    (fin_nntile_time - start_nntile_time) / n_runs))
    elif args.n_fwd > 0:
        print("NNTile timing averaged over {} runs of fwd = {}".format(
                    n_runs,
                    (fin_nntile_time - start_nntile_time) / n_runs))
