# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/model/test_gpt2_model.py
# Test for nntile.model.gpt2
# Each test is generated in float precision by Torch, then it is downcasted
# into NNTile type. So, implementation of double precision is NOT checked.
#
# @version 1.1.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from transformers import GPT2Config as GPT2ConfigTorch
from transformers.models.gpt2.modeling_gpt2 import GPT2Model as GPT2Model_torch

import nntile
from nntile.model.gpt2_config import GPT2ConfigNNTile
from nntile.model.gpt2_model import GPT2Model
from nntile.tensor import to_numpy

# NNTile dtype via corresponding Tensor type
dtype2nntile = {
        'fp32': nntile.tensor.Tensor_fp32,
        'bf16': nntile.tensor.Tensor_bf16,
        'fp32_fast_tf32': nntile.tensor.Tensor_fp32_fast_tf32,
        'fp32_fast_fp16': nntile.tensor.Tensor_fp32_fast_fp16,
        'fp32_fast_bf16': nntile.tensor.Tensor_fp32_fast_bf16,
        'fp16': nntile.tensor.Tensor_fp16,
}

dtype2np = {
        'fp32': np.float32,
        'bf16': np.float16,
        'fp16': np.float32,
}

dtype2tol = {
        'fp32': {'rtol': 1e-5},
        'bf16': {'rtol': 4e-2},
        'fp32_fast_tf32': {'rtol': 2e-3},
        'fp32_fast_fp16': {'rtol': 8e-3},
        'fp32_fast_bf16': {'rtol': 8e-3},
        'fp16': {'rtol': 5e-3},
}

nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda')


@dataclass
class GPT2TestParams:
    vocab_size: int
    vocab_embed_dim_tile: int
    hidden_size: int
    hidden_size_tile: int
    intermediate_size: int
    intermediate_size_tile: int
    batch_size: int
    batch_size_tile: int
    seq_len: int
    seq_len_tile: int
    n_head: int
    n_head_tile: int
    redux: bool = True


single_tile = GPT2TestParams(
    vocab_size=32000,
    vocab_embed_dim_tile=128,
    hidden_size=128,
    hidden_size_tile=128,
    intermediate_size=64,
    intermediate_size_tile=64,
    batch_size=3,
    batch_size_tile=3,
    seq_len=32,
    seq_len_tile=32,
    n_head=16,
    n_head_tile=16
    )

multiple_tiles = GPT2TestParams(
    vocab_size=32000,
    vocab_embed_dim_tile=32,
    hidden_size=128,
    hidden_size_tile=32,
    intermediate_size=64,
    intermediate_size_tile=16,
    batch_size=3,
    batch_size_tile=1,
    seq_len=128,
    seq_len_tile=32,
    n_head=16,
    n_head_tile=8)


def generate_inputs(params: GPT2TestParams,
                    dtype: str,
                    num_hidden_layers: int):
    torch_config = GPT2ConfigTorch(
        vocab_size=params.vocab_size,
        n_embd=params.hidden_size,
        n_layer=num_hidden_layers,
        n_head=params.n_head,
        n_inner=params.intermediate_size,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        scale_attn_weights=True,
        use_cache=False,
        add_cross_attention=False,
    )

    torch_model = GPT2Model_torch(torch_config)
    nntile_config = GPT2ConfigNNTile(
            vocab_size=params.vocab_size,
            vocab_embed_dim_tile=params.vocab_embed_dim_tile,
            hidden_size=params.hidden_size,
            hidden_size_tile=params.hidden_size_tile,
            intermediate_size=params.intermediate_size,
            intermediate_size_tile=params.intermediate_size_tile,
            n_head=params.n_head,
            n_head_tile=params.n_head_tile,
            dtype=dtype,
            num_hidden_layers=num_hidden_layers
    )
    gen = np.random.default_rng(42)

    nntile_model = GPT2Model.from_torch(
            torch_model, params.batch_size, params.batch_size_tile,
            params.seq_len, params.seq_len_tile, nntile_config)
    nntile_model.clear_gradients()
    x_random = gen.integers(params.vocab_size,
                            size=nntile_model.activations[0].value.shape,
                            dtype=np.int64)

    x_nntile = np.array(x_random, dtype=np.int64, order='F')
    nntile_model.activations[0].value.from_array(x_nntile)
    x_torch = torch.tensor(x_nntile.T)
    y_grad_random = gen.standard_normal((params.hidden_size,
                                         params.seq_len,
                                         params.batch_size),
                                        dtype=np.float32)
    y_grad_nntile = np.array(y_grad_random, dtype=np.float32, order='F')
    nntile_model.activations[-1].grad.from_array(y_grad_nntile)
    y_grad_torch = torch.Tensor(y_grad_nntile.T)
    return torch_model, nntile_model, x_torch, y_grad_torch


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
    pytest.param('fp16', marks=nocuda),
])
@pytest.mark.parametrize('num_hidden_layers', [1, 2, 3])
class TestGPT2Model:
    def test_coercion(self, context, torch_rng,
                      params: GPT2TestParams,
                      dtype: str,
                      num_hidden_layers: int):

        torch_model, nntile_model, _, _ = generate_inputs(params, dtype,
                                                        num_hidden_layers)

        torch_model_other = nntile_model.to_torch()
        nntile_model.unregister()
        rtol = dtype2tol[dtype]['rtol']
        for (n1, p1), (n2, p2) in zip(torch_model.named_parameters(),
                torch_model_other.named_parameters()):
            assert n1 == n2
            assert torch.norm(p1 - p2) <= rtol * torch.norm(p1)

    def test_forward(self, context, torch_rng,
                     params: GPT2TestParams,
                     dtype: str,
                     num_hidden_layers: int):
        torch_model, nntile_model, x, _ = generate_inputs(params, dtype,
                                                        num_hidden_layers)
        y = torch_model(x)
        y_torch = y.last_hidden_state
        nntile_model.forward_async()
        y_nntile = torch.Tensor(to_numpy(nntile_model.activations[-1].value).T)
        nntile_model.unregister()

        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(y_torch - y_nntile) <= rtol * torch.norm(y_torch)

    def test_backward(self, context, torch_rng,
                              params: GPT2TestParams,
                              dtype: str,
                              num_hidden_layers: int):
        torch_model, nntile_model, x, y_grad = generate_inputs(params, dtype,
                                                        num_hidden_layers)
        y = torch_model(x)
        nntile_model.forward_async()
        res = (y.last_hidden_state * y_grad).sum()
        res.backward()
        nntile_model.backward_async()
        torch_model_other = nntile_model.to_torch_with_grads()
        nntile_model.unregister()

        rtol = dtype2tol[dtype]['rtol']
        for (n1, p1), (n2, p2) in zip(torch_model.named_parameters(),
                torch_model_other.named_parameters()):
            assert n1 == n2
            assert p1.requires_grad == p2.requires_grad
            if p1.requires_grad:
                g1, g2 = p1.grad, p2.grad
                assert torch.norm(g1 - g2) <= rtol * torch.norm(g1)


@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['fp32', 'fp16', 'bf16'])
def test_bench_gpt2_forward_async(context_cuda, benchmark_model, dtype: str):
    params = single_tile
    num_hidden_layers = 1
    _, nntile_model, _, _ = generate_inputs(params, dtype, num_hidden_layers)

    np_out = np.zeros(
        nntile_model.activations[-1].value.shape, dtype=dtype2np[dtype], order="F"
    )

    def bench_fn():
        nntile_model.forward_async()
        nntile_model.activations[-1].value.to_array(np_out)

    nntile.starpu.wait_for_all()
    benchmark_model(bench_fn)
    nntile_model.unregister()


@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['fp32', 'fp16', 'bf16'])
def test_bench_gpt2_forward_backward_async(context_cuda, benchmark_model, dtype: str):
    params = single_tile
    num_hidden_layers = 1
    _, nntile_model, _, _ = generate_inputs(params, dtype, num_hidden_layers)
    nntile_model.clear_gradients()

    rng = np.random.default_rng(42)
    np_grad = np.array(
        rng.standard_normal(nntile_model.activations[-1].value.shape),
        dtype=dtype2np[dtype],
        order="F",
    )

    def bench_fn():
        nntile_model.forward_async()
        nntile_model.activations[-1].grad.from_array(np_grad)
        nntile_model.backward_async()
        nntile.starpu.wait_for_all()

    nntile.starpu.wait_for_all()
    benchmark_model(bench_fn)
    nntile_model.unregister()
