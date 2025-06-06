# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/model/test_gpt_neo_causal.py
# Test for nntile.model.gpt_neo_causal
# Each test is generated in float precision by Torch, then it is downcasted
# into NNTile type. So, implementation of double precision is NOT checked.
#
# @version 1.1.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers.models.gpt_neo.modeling_gpt_neo import (
    GPTNeoConfig as ConfigTorch, GPTNeoForCausalLM as ModelTorch)

import nntile
from nntile.model.gpt_neo_causal import GPTNeoForCausalLM
from nntile.model.gpt_neo_config import GPTNeoConfig
from nntile.utils.constructors import to_numpy

# NNTile dtype via corresponding Tensor type
dtype2nntile = {
        'fp32': nntile.tensor.Tensor_fp32,
        'fp32_fast_tf32': nntile.tensor.Tensor_fp32_fast_tf32,
        'bf16': nntile.tensor.Tensor_bf16,
        'fp32_fast_fp16': nntile.tensor.Tensor_fp32_fast_fp16,
        'fp32_fast_bf16': nntile.tensor.Tensor_fp32_fast_bf16
}

dtype2tol = {
        'fp32': {'rtol': 8e-6},
        'fp32_fast_tf32': {'rtol': 8e-4},
        'bf16': {'rtol': 1.6e-2},
        'fp32_fast_fp16': {'rtol': 9e-4},
        'fp32_fast_bf16': {'rtol': 4e-3},
}

nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda')


@dataclass
class GPTNeoTestParams:
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


single_tile = GPTNeoTestParams(
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

multiple_tiles = GPTNeoTestParams(
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


def generate_inputs(params: GPTNeoTestParams,
                    dtype: str,
                    attn_pattern: list,
                    pattern_mult: int):
    torch_config = ConfigTorch(
        vocab_size=params.vocab_size,
        hidden_size=params.hidden_size,
        num_layers=len(attn_pattern) * pattern_mult,
        attention_types=[[attn_pattern, pattern_mult]],
        num_heads=params.n_head,
        intermediate_size=params.intermediate_size,
        resid_dropout=0.0,
        embed_dropout=0.0,
        attention_dropout=0.0,
        use_cache=False,
    )
    torch_model = ModelTorch(torch_config)
    torch_model.lm_head.weight = nn.Parameter(
        torch_model.lm_head.weight.detach().clone()
    )
    nntile_config = GPTNeoConfig(
        vocab_size=params.vocab_size,
        vocab_embed_dim_tile=params.vocab_embed_dim_tile,
        hidden_size=params.hidden_size,
        hidden_size_tile=params.hidden_size_tile,
        intermediate_size=params.intermediate_size,
        intermediate_size_tile=params.intermediate_size_tile,
        num_heads=params.n_head,
        num_heads_tile=params.n_head_tile,
        attention_types=[[attn_pattern, pattern_mult]],
        dtype=dtype,
        num_hidden_layers=len(attn_pattern) * pattern_mult,
    )
    gen = np.random.default_rng(42)

    nntile_model = GPTNeoForCausalLM.from_torch(
            torch_model, params.batch_size, params.batch_size_tile,
            params.seq_len, params.seq_len_tile, nntile_config)
    nntile_model.clear_gradients()
    x_random = gen.integers(params.seq_len,
                            size=nntile_model.activations[0].value.shape,
                            dtype=np.int64)

    x_nntile = np.array(x_random, dtype=np.int64, order='F')
    nntile_model.activations[0].value.from_array(x_nntile)
    x_torch = torch.tensor(x_nntile.T)
    y_grad_random = gen.standard_normal((params.vocab_size,
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
])
@pytest.mark.parametrize('attn_pattern',
                         [["global", "local"],
                          ["local", "global"],
                          ["local"]
                          ])
@pytest.mark.parametrize('pattern_mult', [0, 1, 2])
class TestGPTNeoForCausalLM:
    def test_coercion(self, context, torch_rng,
                      params: GPTNeoTestParams,
                      dtype: str,
                      attn_pattern: list,
                      pattern_mult: int):

        torch_model, nntile_model, _, _ = generate_inputs(
            params, dtype, attn_pattern, pattern_mult
        )
        torch_model_other = nntile_model.to_torch()
        nntile_model.unregister()
        rtol = dtype2tol[dtype]['rtol']
        for (n1, p1), (n2, p2) in zip(torch_model.named_parameters(),
                torch_model_other.named_parameters()):
            assert n1 == n2
            assert torch.norm(p1 - p2) <= rtol * torch.norm(p1)

    def test_forward(self, context, torch_rng,
                     params: GPTNeoTestParams,
                     dtype: str,
                     attn_pattern: list,
                     pattern_mult: int):
        torch_model, nntile_model, x, _ = generate_inputs(
            params, dtype, attn_pattern, pattern_mult
        )
        y = torch_model(x, return_dict=True)
        y_torch = y.logits
        nntile_model.forward_async()
        y_nntile = torch.Tensor(
            to_numpy(nntile_model.activations[-1].value).T
        )
        nntile_model.unregister()

        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(y_torch - y_nntile) <= rtol * torch.norm(y_torch)

    def test_forward_backward(self, context, torch_rng,
                              params: GPTNeoTestParams,
                              dtype: str,
                              attn_pattern: list,
                              pattern_mult: int):
        torch_model, nntile_model, x, y_grad = generate_inputs(
            params, dtype, attn_pattern, pattern_mult
        )
        y = torch_model(x, return_dict=True)
        y_torch = y.logits
        nntile_model.forward_async()
        y_nntile = torch.Tensor(
            to_numpy(nntile_model.activations[-1].value).T
        )
        res = (y.logits * y_grad).sum()
        res.backward()
        nntile_model.backward_async()
        torch_model_other = nntile_model.to_torch_with_grads()
        nntile_model.unregister()

        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(y_torch - y_nntile) <= rtol * torch.norm(y_torch)

        for (n1, p1), (n2, p2) in zip(torch_model.named_parameters(),
                torch_model_other.named_parameters()):
            assert n1 == n2
            assert p1.requires_grad == p2.requires_grad
            if p1.requires_grad:
                g1, g2 = p1.grad, p2.grad
                assert torch.norm(g1 - g2) <= rtol * torch.norm(g1)
