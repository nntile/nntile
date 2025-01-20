# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/performance/scaling_layers.py
# Test scaling of layers
#
# @version 1.1.0

import argparse
import json
import time

import numpy as np
import nntile
from nntile.layer import Linear as Linear_nntile
# from nntile.model.llama_causal import LlamaForCausalLM as Llama_nntile
from nntile.tensor import TensorMoments, TensorTraits, notrans

# Create argument parser
parser = argparse.ArgumentParser(prog="Test performance script for layers in NNTile",
        description="")
parser.add_argument("--layer", choices=["linear"],
                    default="linear")

parser.add_argument("--n-fwd", type=int, default=0)
parser.add_argument("--n-fwd-bwd", type=int, default=0)


parser.add_argument("--seq-len", type=int, default=1024)
parser.add_argument("--seq-len-tile", type=int, default=-1)
parser.add_argument("--minibatch-size", type=int, default=1)
parser.add_argument("--minibatch-size-tile", type=int, default=-1)

parser.add_argument("--hidden-size", type=int, default=2)
parser.add_argument("--hidden-size-tile", type=int, default=-1)
parser.add_argument("--n-head", type=int, default=-1)
parser.add_argument("--n-head-tile", type=int, default=-1)

parser.add_argument("--dtype", choices=["fp32", "fp32_fast_tf32", "bf16",
                                        "fp32_fast_fp16", "fp32_fast_bf16"],
                    default="fp32")
parser.add_argument("--restrict", choices=["cpu", "cuda", None],
                    default=None)
parser.add_argument("--flash-attention", action="store_true")
parser.add_argument("--use-redux", action="store_true")
parser.add_argument("--num-warmup-calls", type=int, default=1)
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
assert (args.n_fwd == 0 and args.n_fwd_bwd > 0) or \
       (args.n_fwd > 0 and args.n_fwd_bwd == 0)

dtype2nntile = {
        'fp32': nntile.tensor.Tensor_fp32,
        'fp32_fast_tf32': nntile.tensor.Tensor_fp32_fast_tf32,
        'bf16': nntile.tensor.Tensor_bf16,
        'fp32_fast_fp16': nntile.tensor.Tensor_fp32_fast_fp16,
        'fp32_fast_bf16': nntile.tensor.Tensor_fp32_fast_bf16,
}

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
    nntile.starpu.restrict_cuda()
elif args.restrict == "cpu":
    nntile.starpu.restrict_cpu()

time0 = time.time()
if args.n_head_tile == -1:
    args.n_head_tile = args.n_head
if args.hidden_size_tile == -1:
    args.hidden_size_tile = args.hidden_size

gen = np.random.default_rng(42)
if args.layer == "linear":
    x_shape = [args.hidden_size,
                args.seq_len, args.minibatch_size]
    input_data = gen.standard_normal(x_shape, dtype=np.float32)

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
    nntile_layer, _ = Linear_nntile.generate_simple(X, "R", notrans,
                                                    1, [args.hidden_size],
                                                    [args.hidden_size_tile],
                                                    next_tag, True,
                                                    args.use_redux)
    nntile_layer.init_randn_async()

if args.n_fwd > 0:
    n_runs = args.n_fwd
elif args.n_fwd_bwd > 0:
    n_runs = args.n_fwd_bwd

for n_wup in range(args.num_warmup_calls):
    nntile_layer.forward_async()
    if args.n_fwd_bwd > 0:
        nntile_layer.clear_gradients()
        nntile_layer.y.grad.from_array(
                np.ones(nntile_layer.y.value.shape,
                np.float32, 'F'))
        nntile_layer.backward_async()
    nntile.starpu.wait_for_all()

start_nntile_time = time.time()
nntile.starpu.profiling_enable()
for run_idx in range(n_runs):
    nntile_layer.forward_async()
    if args.n_fwd_bwd > 0:
        nntile_layer.clear_gradients()
        nntile_layer.y.grad.from_array(
                np.ones(nntile_layer.y.value.shape,
                np.float32, 'F'))
        nntile_layer.backward_async()
    nntile.starpu.wait_for_all()
nntile.starpu.profiling_disable()
fin_nntile_time = time.time()

nntile_layer.unregister()
nntile_layer.x.unregister()
nntile_layer.y.unregister()
if args.n_fwd_bwd > 0:
    print("NNTile timing averaged over {} runs of fwd + bwd = {}".format(
                n_runs,
                (fin_nntile_time - start_nntile_time) / n_runs))
elif args.n_fwd > 0:
    print("NNTile timing averaged over {} runs of fwd = {}".format(
                n_runs,
                (fin_nntile_time - start_nntile_time) / n_runs))
