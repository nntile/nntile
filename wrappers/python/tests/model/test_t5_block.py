# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/model/test_t5_block.py
# Test for nntile.model.t5_block - T5Block
#
# @version 1.1.0
# ruff: noqa: E501

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from transformers.models.t5.modeling_t5 import (
    T5Block as T5BlockTorch, T5Config as T5ConfigTorch)

import nntile
import nntile.utils.constructors as nntc
from nntile.model.t5_block import T5Block
from nntile.model.t5_config import T5ConfigNNTile
from nntile.tensor import TensorMoments, TensorTraits

# NNTile dtype via corresponding Tensor type
dtype2nntile = {
    "fp32": nntile.tensor.Tensor_fp32,
    "fp32_fast_tf32": nntile.tensor.Tensor_fp32_fast_tf32,
    "bf16": nntile.tensor.Tensor_bf16,
    "fp16": nntile.tensor.Tensor_fp16,
}

dtype2np = {
    "fp32": np.float32,
    "bf16": np.float16,
    "fp16": np.float32,
}

dtype2tol = {
    "fp32": {"rtol": 4.5e-5},
    "fp32_fast_tf32": {"rtol": 7e-4},
    "bf16": {"rtol": 1.2e-2},
}

nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")


@dataclass
class T5BlockTestParams:
    d_model: int
    d_model_tile: int
    d_kv: int
    d_kv_tile: int
    d_ff: int
    d_ff_tile: int
    n_head: int
    n_head_tile: int
    n_batch: int
    n_batch_tile: int
    is_decoder: bool = False
    redux: bool = False  # Disabled because it causes SegFaults
    seq_len: int = 100
    seq_len_tile: int = 100
    is_gated_act: bool = True


encoder_single_tile = T5BlockTestParams(
    d_model=128,
    d_model_tile=128,
    d_kv=64,
    d_kv_tile=64,
    d_ff=128,
    d_ff_tile=128,
    n_head=4,
    n_head_tile=4,
    seq_len=64,
    seq_len_tile=64,
    n_batch=1,
    n_batch_tile=1,
    is_decoder=False,
    is_gated_act=True,
)

decoder_single_tile = T5BlockTestParams(
    d_model=128,
    d_model_tile=128,
    d_kv=64,
    d_kv_tile=64,
    d_ff=128,
    d_ff_tile=128,
    n_head=4,
    n_head_tile=4,
    seq_len=64,
    seq_len_tile=64,
    n_batch=1,
    n_batch_tile=1,
    is_decoder=True,
    is_gated_act=True,
)

multiple_tiles = T5BlockTestParams(
    d_model=128,
    d_model_tile=32,
    d_kv=64,
    d_kv_tile=16,
    d_ff=128,
    d_ff_tile=32,
    n_head=4,
    n_head_tile=2,
    seq_len=64,
    seq_len_tile=16,
    n_batch=4,
    n_batch_tile=1,
    is_decoder=False,
    is_gated_act=True,
)


def generate_inputs(params: T5BlockTestParams, dtype: str):
    # Configure PyTorch T5 layer
    torch_config = T5ConfigTorch(
        d_model=params.d_model,
        d_ff=params.d_ff,
        d_kv=params.d_kv,
        num_heads=params.n_head,
        dropout_rate=0.0,
        dense_act_fn="gelu_new",
        is_decoder=params.is_decoder,
        is_gated_act=params.is_gated_act,
        attn_implementation="eager",
    )
    torch_block = T5BlockTorch(torch_config)

    # Configure NNTile T5 layer
    nntile_config = T5ConfigNNTile(
        d_model=params.d_model,
        d_model_tile=params.d_model_tile,
        d_kv=params.d_kv,
        d_kv_tile=params.d_kv_tile,
        d_ff=params.d_ff,
        d_ff_tile=params.d_ff_tile,
        n_head=params.n_head,
        n_head_tile=params.n_head_tile,
        redux=params.redux,
        is_decoder=params.is_decoder,
        is_gated_act=params.is_gated_act,
    )

    # Make sure all dropout layers are disabled
    for name, module in torch_block.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

    torch_block.eval()  # Set to evaluation mode

    # Set input tensor dimensions
    x_shape = [params.d_model, params.seq_len, params.n_batch]
    x_basetile = [params.d_model_tile, params.seq_len_tile, params.n_batch_tile]
    x_traits = TensorTraits(x_shape, x_basetile)
    x_distr = [0] * x_traits.grid.nelems
    x_type = dtype2nntile[dtype]
    x_value = x_type(x_traits, x_distr)
    x_grad = x_type(x_traits, x_distr)
    X = TensorMoments(x_value, x_grad, grad_required=True)

    # Generate random input data
    gen = np.random.default_rng(42)
    x_random = gen.standard_normal(x_shape, dtype=np.float32)
    x_nntile = np.array(x_random, dtype=np.float32, order="F")
    x_value.from_array(x_nntile)
    x_torch = torch.Tensor(x_nntile.T)
    x_torch.requires_grad_()

    eo_torch, eo_nnt = None, None
    if params.is_decoder:
        eo_random = gen.standard_normal(x_shape, dtype=np.float32)
        eo_nntile = np.array(eo_random, dtype=np.float32, order="F")
        eo_torch = torch.Tensor(eo_nntile.T)
        eo_nnt = TensorMoments(
            nntc.from_array(eo_random, x_basetile),
            nntc.zeros(eo_random.shape, x_basetile),
            True,
        )

    # Initialize NNTile layer from PyTorch layer
    nntile_block = T5Block.from_torch(
        torch_block, X, nntile_config, encoder_output=eo_nnt
    )
    nntile_block.clear_gradients()

    # Generate random gradient for backward pass
    y_grad_random = gen.standard_normal(x_shape, dtype=np.float32)
    y_grad_nntile = np.array(y_grad_random, dtype=np.float32, order="F")
    nntile_block.activations[-1].grad.from_array(y_grad_nntile)
    y_grad_torch = torch.Tensor(y_grad_nntile.T)

    return torch_block, nntile_block, x_torch, y_grad_torch, eo_torch, eo_nnt


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(encoder_single_tile, id="encoder_single_tile"),
        pytest.param(decoder_single_tile, id="decoder_single_tile"),
        pytest.param(multiple_tiles, id="multiple_tiles"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        "fp32",
        # pytest.param("fp32_fast_tf32", marks=nocuda),
        # pytest.param("bf16", marks=nocuda),
    ],
)
class TestT5Block:
    def test_forward(
        self, context, torch_rng, params: T5BlockTestParams, dtype: str
    ):
        """Test that forward pass gives same results in PyTorch and NNTile"""
        torch_block, nntile_block, x, _, eo_torch, _eo_nnt = generate_inputs(
            params, dtype
        )

        # PyTorch forward pass
        cache_position = torch.arange(x.shape[1], dtype=torch.long, device=x.device)
        y = torch_block(x, encoder_hidden_states=eo_torch, cache_position=cache_position)[0]

        # NNTile forward pass
        nntile_block.forward_async()
        y_nntile = torch.Tensor(nntc.to_numpy(nntile_block.activations[-1].value).T)
        nntile_block.unregister()

        # Compare results
        rtol = dtype2tol[dtype]["rtol"]
        assert torch.norm(y - y_nntile) <= rtol * torch.norm(y)

    def test_backward(
        self, context, torch_rng, params: T5BlockTestParams, dtype: str
    ):
        """Test that backward pass gives same results in PyTorch and NNTile"""
        torch_block, nntile_block, x, y_grad, eo_torch, _eo_nnt = generate_inputs(
            params, dtype
        )

        # PyTorch forward and backward pass
        cache_position = torch.arange(x.shape[1], dtype=torch.long, device=x.device)
        y = torch_block(x, encoder_hidden_states=eo_torch, cache_position=cache_position)[0]
        res = (y * y_grad).sum()
        res.backward()

        # NNTile forward and backward pass
        nntile_block.forward_async()
        nntile_block.backward_async()

        # Compare gradients
        grad_nntile = torch.Tensor(nntc.to_numpy(nntile_block.activations[0].grad).T)
        rtol = dtype2tol[dtype]["rtol"]

        assert torch.norm(x.grad - grad_nntile) <= rtol * torch.norm(x.grad)

        # Clean up
        nntile_block.unregister()

@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['fp32', 'fp16', 'bf16'])
def test_bench_t5_block_forward_async(context_cuda, benchmark_model, dtype: str):
    if dtype == 'fp16':
        pytest.xfail("not supported")

    params = encoder_single_tile
    _, nntile_block, *_ = generate_inputs(params, dtype)

    def bench_fn():
        nntile_block.forward_async()
        nntile.starpu.wait_for_all()

    nntile.starpu.wait_for_all()
    benchmark_model(bench_fn)
    nntile_block.unregister()


@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['fp32', 'fp16', 'bf16'])
def test_bench_t5_block_forward_backward_async(context_cuda, benchmark_model, dtype: str):
    if dtype == 'fp16':
        pytest.xfail("not supported")

    params = encoder_single_tile
    _, nntile_block, *_ = generate_inputs(params, dtype)

    def bench_fn():
        nntile_block.forward_async()
        nntile_block.backward_async()
        nntile.starpu.wait_for_all()

    nntile.starpu.wait_for_all()
    benchmark_model(bench_fn)
    nntile_block.unregister()
