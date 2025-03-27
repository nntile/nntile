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
import os
import pathlib

import numpy as np
import torch
from transformers import LlamaConfig
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import (
    LlamaAttention, LlamaDecoderLayer, LlamaMLP, LlamaRotaryEmbedding)

import nntile
from nntile.layer.llama_attention import (
    LlamaAttention as LlamaAttention_nntile)
from nntile.model.llama_causal import LlamaForCausalLM as Llama_nntile
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
                                            "attention", "causal-llama"],
                    default="mlp")

parser.add_argument("--attn-implementation",
                    choices=["eager", "sdpa", "flash_attention_2"],
                    default="eager")

parser.add_argument("--use-torch", action="store_true")
parser.add_argument("--use-nntile", action="store_true")

parser.add_argument("--n-iters", type=int, default=0)
parser.add_argument("--mode", choices=["fwd", "fwd-bwd"], default="fwd")

parser.add_argument("--seq-len", type=int, default=1024)
parser.add_argument("--seq-len-tile", type=int, default=-1)
parser.add_argument("--minibatch-size", type=int, default=1)
parser.add_argument("--minibatch-size-tile", type=int, default=-1)

parser.add_argument("--hidden-size", type=int, default=-1)
parser.add_argument("--hidden-size-tile", type=int, default=-1)
parser.add_argument("--intermediate-size", type=int, default=-1)
parser.add_argument("--intermediate-size-tile", type=int, default=-1)
parser.add_argument("--n-head-tile", type=int, default=-1)
parser.add_argument("--head-dim", type=int, default=-1)
parser.add_argument("--kv-heads-ratio", type=int, default=1)

parser.add_argument("--dtype", choices=["fp32", "fp32_fast_tf32", "bf16",
                                        "fp32_fast_fp16", "fp32_fast_bf16"],
                    default="fp32")
parser.add_argument("--restrict", choices=["cpu", "cuda", None],
                    default=None)
parser.add_argument("--flash-attention", action="store_true")
parser.add_argument("--use-redux", action="store_true")
parser.add_argument("--torch-compile", action="store_true")
parser.add_argument("--num-warmup-calls", type=int, default=1)
parser.add_argument("--logger", action="store_true")
parser.add_argument("--logger-server-addr", type=str,
                    default="localhost")
parser.add_argument("--logger-server-port", type=int, default=5001)
parser.add_argument("--results-folder", type=str, default=".results")

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

if not os.path.isdir(args.results_folder):
    path2res_filder = pathlib.Path(args.results_folder)
    pathlib.Path.mkdir(path2res_filder, parents=True)

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

if args.hidden_size != -1:
    llama_torch_config.hidden_size = args.hidden_size
    assert args.hidden_size % llama_torch_config.num_attention_heads == 0
    llama_torch_config.head_dim = args.hidden_size // llama_torch_config.num_attention_heads

if args.hidden_size != -1 and args.head_dim != -1:
    llama_torch_config.head_dim = args.head_dim
    llama_torch_config.num_attention_heads = args.hidden_size // args.head_dim
    assert llama_torch_config.num_attention_heads % args.kv_heads_ratio == 0
    llama_torch_config.num_key_value_heads = llama_torch_config.num_attention_heads // args.kv_heads_ratio

if args.intermediate_size != -1:
    llama_torch_config.intermediate_size = args.intermediate_size
# print(llama_torch_config)

if args.use_nntile:
    # Initialize NNTile and StarPU
    time0 = time.time()
    # Set up StarPU+MPI and init codelets
    nntile_config = nntile.starpu.Config(-1, -1, 1, args.logger,
            args.logger_server_addr, args.logger_server_port)
    nntile.starpu.profiling_init()
    nntile.starpu.profiling_disable()
    nntile.starpu.init()
    time1 = time.time() - time0
    print("StarPU + NNTile + MPI init in {} seconds".format(time1))
    next_tag = 0
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
    x_torch = torch.tensor(np.array(input_data,
                                    dtype=np.float32,
                                    order="F").T,
                                    requires_grad=True)
    if args.use_nntile:
        x_nntile = np.array(input_data, dtype=np.float32, order="F")
        x_type = dtype2nntile[args.dtype]

        x_basetile = [args.hidden_size_tile,
                    args.seq_len_tile,
                    args.minibatch_size_tile]
        x_traits = TensorTraits(x_shape, x_basetile)
        x_distr = [0] * x_traits.grid.nelems
        x_value = x_type(x_traits, x_distr, 0)
        x_value.from_array(x_nntile)
        x_grad = x_type(x_traits, x_distr, 0)
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
        rotary_emb = LlamaRotaryEmbedding(config=llama_torch_config).to(torch_device)
        pos_embs = rotary_emb(torch_layer_.self_attn.v_proj.weight,
                                    pos_ids_torch)
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
        x_distr = [0] * x_traits.grid.nelems
        x_type = dtype2nntile[args.dtype]
        x_value = x_type(x_traits, x_distr, 0)
        x_grad = x_type(x_traits, x_distr, 0)
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
        torch_layer_ = torch_layer_.to(torch_device)
        mask_torch = torch.Tensor(np.array(1 - mask, dtype=np.float32)).T \
                * torch.finfo(torch.float32).min
        mask_torch = mask_torch[None, None, :, :].expand(args.minibatch_size,
                                                1, -1, -1).to(torch_device)
        pos_ids_torch = torch.tensor(pos_ids, dtype=torch.long).to(torch_device)
        rotary_emb = LlamaRotaryEmbedding(config=llama_torch_config).to(torch_device)
        pos_embs = rotary_emb(torch_layer_.v_proj.weight,
                                    pos_ids_torch)

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
        x_distr = [0] * x_traits.grid.nelems
        x_type = dtype2nntile[args.dtype]
        x_value = x_type(x_traits, x_distr, 0)
        x_grad = x_type(x_traits, x_distr, 0)
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
elif args.submodule == "causal-llama":
    torch_layer_ = LlamaForCausalLM(llama_torch_config)
    mask = np.array(np.triu(np.ones((args.seq_len, args.seq_len))),
                        dtype=bool, order="F")
    gen = np.random.default_rng(42)
    pos_ids = gen.integers(args.seq_len,
                            size=(args.minibatch_size, args.seq_len),
                            dtype=np.int64)
    if args.use_torch:
        # mask_torch = torch.Tensor(np.array(1 - mask, dtype=np.float32)).T \
        #         * torch.finfo(torch.float32).min
        # mask_torch = mask_torch[None, None, :, :].expand(args.minibatch_size,
        #                                         1, -1, -1).to(torch_device)
        pos_ids_torch = torch.tensor(pos_ids).to(torch_device)
        # rotary_emb = LlamaRotaryEmbedding(config=llama_torch_config).to(torch_device)
        # pos_embs = rotary_emb(torch_layer_.self_attn.v_proj.weight,
        #                             pos_ids_torch)

    x_shape = [args.seq_len, args.minibatch_size]
    x_random = gen.integers(args.seq_len, size=x_shape, dtype=np.int64)
    x_torch = torch.tensor(np.array(x_random, order="F").T,
                           requires_grad=False)
    if args.use_nntile:

        x_basetile = [
                    args.seq_len_tile,
                    args.minibatch_size_tile]
        x_traits = TensorTraits(x_shape, x_basetile)
        x_distr = [0] * x_traits.grid.nelems
        x_type = dtype2nntile[args.dtype]
        x_nntile = np.array(x_random, dtype=np.int64, order="F")
        time0 = time.time()
        nntile_module, _ = Llama_nntile.from_torch(torch_layer_,
                                                   args.minibatch_size,
                                                   args.minibatch_size_tile,
                                                   args.seq_len,
                                                   args.seq_len_tile,
                                                   pos_ids, mask,
                                                   llama_config_nntile, 0)
        nntile_module.activations[0].value.from_array(x_nntile)
        time1 = time.time() - time0
        print("Converting PyTorch model to NNTile requires ",
            "{} seconds".format(time1))
        del torch_layer_

    # raise ValueError("Causal LLaMa is not supported yet!")

if args.use_torch and args.torch_compile:
    torch_layer = torch.compile(torch_layer_)
elif args.use_torch:
    torch_layer = torch_layer_

timings = []
if args.use_torch:
    if args.dtype == "bf16":
        torch_layer = torch_layer.bfloat16()
        x_torch = x_torch.bfloat16()
        if args.submodule in ("decoder", "attention"):
            pos_ids_torch = pos_ids_torch.bfloat16()
            mask_torch = mask_torch.bfloat16()
            pos_embs = pos_embs.bfloat16()
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
                                position_embeddings=pos_embs,
                                position_ids=pos_ids_torch,
                                attention_mask=mask_torch)[0]
        elif args.submodule == "attention":
            output = torch_layer(x_torch, position_embeddings=pos_embs,
                            position_ids=pos_ids_torch,
                            attention_mask=mask_torch)[0]
        elif args.submodule == "causal-llama":
            output = torch_layer(x_torch,
                                position_ids=pos_ids_torch).logits
        if args.mode == "fwd-bwd":
            loss = torch.sum(output)
            loss.backward()

    if torch_device == "cuda":
        torch.cuda.synchronize()
    for n_fwd_idx in range(args.n_iters):
        start_torch_time = time.time()
        if args.submodule == "mlp":
            output = torch_layer(x_torch)
        elif args.submodule == "decoder":
            output = torch_layer(x_torch,
                                 position_embeddings=pos_embs,
                                 position_ids=pos_ids_torch,
                                 attention_mask=mask_torch)[0]
        elif args.submodule == "attention":
            output = torch_layer(x_torch, position_embeddings=pos_embs,
                            position_ids=pos_ids_torch,
                            attention_mask=mask_torch)[0]
        elif args.submodule == "causal-llama":
            output = torch_layer(x_torch,
                                 position_ids=pos_ids_torch,
                                 return_dict=True).logits
        if args.mode == "fwd-bwd":
            loss = torch.sum(output)
            loss.backward()
            torch_layer.zero_grad()

        if torch_device == "cuda":
            torch.cuda.synchronize()
        timings.append(time.time() - start_torch_time)
    if args.mode == "fwd-bwd":
        print("PyTorch timing averaged over {} runs fwd + bwd = {}".format(
                                args.n_iters, np.mean(np.array(timings))))
    elif args.mode == "fwd":
        print("PyTorch timing averaged over {} runs of only fwd = {}".format(
                                args.n_iters, np.mean(np.array(timings))))

if args.use_nntile:

    for n_wup in range(args.num_warmup_calls):
        nntile_module.forward_async()
        if args.mode == "fwd-bwd":
            nntile_module.clear_gradients()
            if args.submodule in ("mlp", "decoder", "causal-llama"):
                nntile_module.activations[-1].grad.from_array(
                    np.ones(nntile_module.activations[-1].value.shape,
                    np.float32, 'F'))
            elif args.submodule == "attention":
                nntile_module.y.grad.from_array(
                    np.ones(nntile_module.y.value.shape,
                    np.float32, 'F'))
            nntile_module.backward_async()
        nntile.starpu.wait_for_all()

    nntile.starpu.profiling_enable()
    for run_idx in range(args.n_iters):
        start_nntile_time = time.time()
        nntile_module.forward_async()
        if args.mode == "fwd-bwd":
            nntile_module.clear_gradients()
            if args.submodule in ("mlp", "decoder", "causal-llama"):
                nntile_module.activations[-1].grad.from_array(
                    np.ones(nntile_module.activations[-1].value.shape,
                    np.float32, 'F'))
            elif args.submodule == "attention":
                nntile_module.y.grad.from_array(
                    np.ones(nntile_module.y.value.shape,
                    np.float32, 'F'))
            nntile_module.backward_async()
        nntile.starpu.wait_for_all()
        timings.append(time.time() - start_nntile_time)
    nntile.starpu.profiling_disable()

    nntile_module.unregister()
    if args.submodule == "attention":
        nntile_module.x.unregister()
        nntile_module.y.unregister()
    if args.mode == "fwd-bwd":
        print("NNTile timing averaged over {} runs of fwd + bwd = {}".format(
                    args.n_iters,
                    np.mean(np.array(timings))))
    elif args.mode == "fwd":
        print("NNTile timing averaged over {} runs of fwd = {}".format(
                    args.n_iters,
                    np.mean(np.array(timings))))
if args.use_nntile:
    backend = "nntile"
elif args.use_torch and args.torch_compile == False:
    backend = "torch"
elif args.use_torch and args.torch_compile:
    backend = "torch-compile"

filename = "hsizetile_{}_intermsizetile_{}".format(args.hidden_size_tile,
                                                    args.intermediate_size_tile)
# if args.seq_len_tile != -1:
#     filename = filename + "seqlentile_" + str(args.seq_len_tile)

# if args.hidden_size_tile != -1:
#     filename = filename + "hsizetile_" + str(args.hidden_size_tile)

# np.savez(args.results_folder + "/" + filename, timings=timings,
#          hidden_size=llama_torch_config.hidden_size)

np.savez(args.results_folder + "/" + filename, timings=timings,
         hsize=args.hidden_size, seqlen_tile=args.hidden_size_tile, args=args)