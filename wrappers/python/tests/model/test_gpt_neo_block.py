# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_gpt_neo_block.py
# Test for nntile.model.gpt_neo_block
# Each test is generated in float precision by PyTorch, then it is downcasted
# into NNTile type. So, implementation of double precision is NOT checked.
#
# @version 1.1.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from transformers.models.gpt_neo.modeling_gpt_neo import (
    GPTNeoBlock as GPTNeoBlockTorch, GPTNeoConfig as GPTNeoConfigTorch)

import nntile
from nntile.model.gpt_neo_block import GPTNeoBlock
from nntile.model.gpt_neo_config import GPTNeoConfig
from nntile.tensor import TensorMoments, TensorTraits
from nntile.utils.constructors import to_numpy

# NNTile dtype via corresponding Tensor type
dtype2nntile = {
        'fp32': nntile.tensor.Tensor_fp32,
        'fp32_fast_tf32': nntile.tensor.Tensor_fp32_fast_tf32,
        'bf16': nntile.tensor.Tensor_bf16,
        'fp32_fast_fp16': nntile.tensor.Tensor_fp32_fast_fp16,
        'fp32_fast_bf16': nntile.tensor.Tensor_fp32_fast_bf16
}

dtype2np = {
        'fp32': np.float32,
        'bf16': np.float16,
        'fp16': np.float16,
}

dtype2tol = {
        'fp32': {'rtol': 1e-6},
        'fp32_fast_tf32': {'rtol': 8e-4},
        'bf16': {'rtol': 1.6e-2},
        'fp32_fast_fp16': {'rtol': 8e-4},
        'fp32_fast_bf16': {'rtol': 7e-3},
}

nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda')


@dataclass
class GPTNeoBlockTestParams:
    hidden_size: int
    hidden_size_tile: int
    intermediate_size: int
    intermediate_size_tile: int
    n_batch: int
    n_batch_tile: int
    redux: bool = True
    seq_len: int = 100
    seq_len_tile: int = 100
    num_heads: int = 16
    num_heads_tile: int = 16


single_tile = GPTNeoBlockTestParams(
    hidden_size=128,
    hidden_size_tile=128,
    intermediate_size=64,
    intermediate_size_tile=64,
    seq_len=32,
    seq_len_tile=32,
    n_batch=3,
    n_batch_tile=3)

multiple_tiles = GPTNeoBlockTestParams(
    hidden_size=128,
    hidden_size_tile=32,
    intermediate_size=64,
    intermediate_size_tile=16,
    seq_len=128,
    seq_len_tile=32,
    n_batch=4,
    n_batch_tile=1)


def generate_inputs(params: GPTNeoBlockTestParams,
                    layer_id: int,
                    dtype: str):
    torch_layer_config = GPTNeoConfigTorch(
        hidden_size=params.hidden_size,
        num_heads=params.num_heads,
        intermediate_size=params.intermediate_size,
        resid_dropout=0.0,
        embed_dropout=0.0,
        attention_dropout=0.0,
        use_cache=False,
    )
    torch_module = GPTNeoBlockTorch(torch_layer_config, layer_id)
    nntile_config = GPTNeoConfig(
        vocab_size=torch_layer_config.vocab_size,
        vocab_embed_dim_tile=params.hidden_size,
        hidden_size=params.hidden_size,
        hidden_size_tile=params.hidden_size_tile,
        intermediate_size=params.intermediate_size,
        intermediate_size_tile=params.intermediate_size_tile,
        num_heads=params.num_heads,
        num_heads_tile=params.num_heads_tile,
        attention_types=torch_layer_config.attention_types,
        dtype=dtype
    )
    x_shape = [params.hidden_size, params.seq_len, params.n_batch]
    x_basetile = [params.hidden_size_tile,
                  params.seq_len_tile,
                  params.n_batch_tile]
    x_traits = TensorTraits(x_shape, x_basetile)
    x_distr = [0] * x_traits.grid.nelems
    x_type = dtype2nntile[dtype]
    x_value = x_type(x_traits, x_distr)
    x_grad = x_type(x_traits, x_distr)
    X = TensorMoments(x_value, x_grad, grad_required=True)
    gen = np.random.default_rng(42)
    x_random = gen.standard_normal(x_shape, dtype=np.float32)
    x_nntile = np.array(x_random, dtype=np.float32, order="F")
    x_value.from_array(x_nntile)
    x_torch = torch.Tensor(x_nntile.T)
    x_torch.requires_grad_()
    nntile_module = GPTNeoBlock.from_torch(torch_module, X,
                                                     nntile_config)
    nntile_module.clear_gradients()
    y_grad_random = gen.standard_normal(x_shape, dtype=np.float32)
    y_grad_nntile = np.array(y_grad_random, dtype=np.float32, order="F")
    nntile_module.activations[-1].grad.from_array(y_grad_nntile)
    y_grad_torch = torch.Tensor(y_grad_nntile.T)
    return torch_module, nntile_module, x_torch, y_grad_torch


@pytest.mark.parametrize('params', [
    pytest.param(single_tile, id='single_tile'),
    pytest.param(multiple_tiles, id='multiple_tiles'),
])
@pytest.mark.parametrize('dtype', [
    'fp32',
    pytest.param('bf16', marks=nocuda),
    pytest.param('fp32_fast_tf32', marks=nocuda),
    pytest.param('fp32_fast_fp16', marks=nocuda),
    pytest.param('fp32_fast_bf16', marks=nocuda),
])
@pytest.mark.parametrize('layer_id', [
    pytest.param(1, id='odd_layer_id'),
    pytest.param(2, id='even_layer_id'),
])
class TestGPTNeoBlock:
    def test_coercion(self, context, torch_rng, layer_id: int,
                      params: GPTNeoBlockTestParams, dtype: str):
        torch_module, nntile_layer, *_ = generate_inputs(
            params, layer_id, dtype
        )
        nntile2torch_module = nntile_layer.to_torch()
        nntile_layer.unregister()

        rtol = dtype2tol[dtype]['rtol']
        for (n1, p1), (n2, p2) in zip(torch_module.named_parameters(),
                nntile2torch_module.named_parameters()):
            assert n1 == n2
            assert torch.norm(p1 - p2) <= rtol * torch.norm(p1)

    def test_forward(self, context, torch_rng, layer_id: int,
                     params: GPTNeoBlockTestParams, dtype: str):
        torch_module, nntile_module, x, _ = generate_inputs(
            params, layer_id, dtype
        )
        y = torch_module(x)[0]
        nntile_module.forward_async()
        y_nntile = torch.Tensor(
            to_numpy(nntile_module.activations[-1].value).T
        )
        nntile_module.unregister()
        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(y - y_nntile) <= rtol * torch.norm(y)

    def test_backward(self, context, torch_rng, layer_id: int,
                      params: GPTNeoBlockTestParams, dtype: str):
        torch_module, nntile_module, x, y_grad = generate_inputs(
            params, layer_id, dtype
        )
        nntile2torch_module = nntile_module.to_torch()
        y = torch_module(x)[0]
        nntile_module.forward_async()
        res = (y * y_grad).sum()
        res.backward()
        nntile_module.backward_async()
        nntile2torch_module = nntile_module.to_torch_with_grads()
        grad_nntile = torch.Tensor(
            to_numpy(nntile_module.activations[0].grad).T
        )
        nntile_module.unregister()
        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(x.grad - grad_nntile) <= rtol * torch.norm(x.grad)

        for (n1, p1), (n2, p2) in zip(torch_module.named_parameters(),
                nntile2torch_module.named_parameters()):
            assert n1 == n2
            assert p1.requires_grad == p2.requires_grad
            if p1.requires_grad:
                g1, g2 = p1.grad, p2.grad
                assert torch.norm(g1 - g2) <= rtol * torch.norm(g1)

@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['fp32', 'bf16'])
def test_bench_gpt_neo_block_forward_async(context_cuda, benchmark_model, dtype: str):
    params = single_tile
    _, nntile_module, *_ = generate_inputs(params, layer_id=1, dtype=dtype)

    np_out = np.zeros(
        nntile_module.activations[-1].value.shape, dtype=dtype2np[dtype], order="F"
    )

    def bench_fn():
        nntile_module.forward_async()
        nntile_module.activations[-1].value.to_array(np_out)

    nntile.starpu.wait_for_all()
    benchmark_model(bench_fn)
    nntile_module.unregister()


@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['fp32', 'bf16'])
def test_bench_gpt_neo_block_backward_async(context_cuda, benchmark_model, dtype: str):
    params = single_tile
    _, nntile_module, *_ = generate_inputs(params, layer_id=1, dtype=dtype)

    rng = np.random.default_rng(42)
    np_grad = np.array(
        rng.standard_normal(nntile_module.activations[-1].value.shape),
        dtype=dtype2np[dtype],
        order="F",
    )

    def bench_fn():
        nntile_module.clear_gradients()
        nntile_module.forward_async()
        nntile_module.activations[-1].grad.from_array(np_grad)
        nntile_module.backward_async()
        nntile.starpu.wait_for_all()

    nntile.starpu.wait_for_all()
    benchmark_model(bench_fn)
    nntile_module.unregister()

