# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/model/test_llama_causal.py
# Test for nntile.model.llama_causal
# Each test is generated in float precision by Torch, then it is downcasted
# into NNTile type. So, implementation of double precision is NOT checked.
#
# @version 1.1.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from transformers.models.llama import (
    LlamaConfig as LlamaConfig_torch, LlamaForCausalLM as LlamaModel_torch)

import nntile
from nntile.model.llama_causal import LlamaForCausalLM as LlamaModel_nntile
from nntile.model.llama_config import LlamaConfigNNTile
from nntile.tensor import to_numpy

# NNTile dtype via corresponding Tensor type
dtype2nntile = {
        'fp32': nntile.tensor.Tensor_fp32,
        'fp32_fast_tf32': nntile.tensor.Tensor_fp32_fast_tf32,
        'bf16': nntile.tensor.Tensor_bf16,
}

dtype2tol = {
        'fp32': {'rtol': 1e-5},
        'fp32_fast_tf32': {'rtol': 2e-3},
        'bf16': {'rtol': 4e-2},
}

nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda')


@dataclass
class LlamaTestParams:
    vocab_size: int
    vocab_embed_dim_tile: int
    hidden_size: int
    hidden_size_tile: int
    max_position_embeddings: int
    intermediate_size: int
    intermediate_size_tile: int
    rms_norm_eps: float
    num_attention_heads: int
    num_attention_heads_tile: int
    num_key_value_heads: int
    activation_function: str = "silu"
    attention_dropout: float = 0.0
    rope_theta: float = 10000.
    seq_len: int = 1
    seq_len_tile: int = 1
    batch_size: int = 1
    batch_size_tile: int = 1
    flash_attention: bool = False
    redux: bool = False


multiple_tiles = LlamaTestParams(
            vocab_size=32000,
            vocab_embed_dim_tile=32,
            hidden_size=128,
            hidden_size_tile=32,
            max_position_embeddings=1024,
            intermediate_size=384,
            intermediate_size_tile=96,
            rms_norm_eps=1e-6,
            num_attention_heads=16,
            num_attention_heads_tile=8,
            num_key_value_heads=4,
            activation_function="silu",
            attention_dropout=0.0,
            rope_theta=2.,
            seq_len=64,
            seq_len_tile=16,
            batch_size=4,
            batch_size_tile=1,
            flash_attention=False,
            redux=False)

single_tile = LlamaTestParams(
            vocab_size=32000,
            vocab_embed_dim_tile=128,
            hidden_size=128,
            hidden_size_tile=128,
            max_position_embeddings=1024,
            intermediate_size=384,
            intermediate_size_tile=384,
            rms_norm_eps=1e-6,
            num_attention_heads=16,
            num_attention_heads_tile=16,
            num_key_value_heads=4,
            activation_function="silu",
            attention_dropout=0.0,
            rope_theta=2.,
            seq_len=64,
            seq_len_tile=64,
            batch_size=4,
            batch_size_tile=4,
            flash_attention=False,
            redux=False)


def generate_inputs(params: LlamaTestParams,
                    dtype: str,
                    num_hidden_layers: int,
                    att_bias: bool):
    torch_config = LlamaConfig_torch(
            vocab_size=params.vocab_size,
            hidden_size=params.hidden_size,
            max_position_embeddings=params.max_position_embeddings,
            intermediate_size=params.intermediate_size,
            num_attention_heads=params.num_attention_heads,
            num_key_value_heads=params.num_key_value_heads,
            attention_bias=att_bias,
            use_cache=False,
            attention_dropout=0.0,
            num_hidden_layers=num_hidden_layers,
            rms_norm_eps=params.rms_norm_eps
    )

    torch_model = LlamaModel_torch(
            torch_config,
    )
    nntile_config = LlamaConfigNNTile(
            vocab_size=params.vocab_size,
            vocab_embed_dim_tile=params.hidden_size_tile,
            hidden_size=params.hidden_size,
            hidden_size_tile=params.hidden_size_tile,
            intermediate_size=params.intermediate_size,
            intermediate_size_tile=params.intermediate_size_tile,
            num_hidden_layers=num_hidden_layers,
            rms_norm_eps=params.rms_norm_eps,
            max_position_embeddings=torch_config.max_position_embeddings,
            n_attention_head=torch_config.num_attention_heads,
            n_head_tile=torch_config.num_attention_heads,
            num_key_value_heads=torch_config.num_key_value_heads,
            dtype=dtype,
            attention_bias=att_bias,
            flash_attention=params.flash_attention
    )
    gen = np.random.default_rng(42)
    pos_ids = gen.integers(params.seq_len,
                           size=(params.batch_size, params.seq_len),
                           dtype=np.int64)
    mask = np.array(np.triu(np.ones((params.seq_len, params.seq_len))),
                    dtype=bool, order="F")
    nntile_model, _ = LlamaModel_nntile.from_torch(
            torch_model, params.batch_size, params.batch_size_tile,
            params.seq_len, params.seq_len_tile, pos_ids,
            mask, nntile_config, 0)
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
    return torch_model, nntile_model, x_torch, pos_ids, y_grad_torch


@pytest.mark.parametrize('params', [
    pytest.param(single_tile, id='single_tile'),
    pytest.param(multiple_tiles, id='multiple_tiles'),
])
@pytest.mark.parametrize('dtype', [
    'fp32',
    pytest.param('fp32_fast_tf32', marks=nocuda),
    pytest.param('bf16', marks=nocuda),
])
@pytest.mark.parametrize('num_hidden_layers', [1, 2, 3])
@pytest.mark.parametrize('att_bias', [
    False,
    # True # Temporarily disabled to investigate later
])
class TestLlama:
    def test_coercion(self, starpu_simple, torch_rng,
                      params: LlamaTestParams,
                      dtype: str,
                      num_hidden_layers: int,
                      att_bias: bool):

        torch_model, nntile_model, *_ = generate_inputs(params,
                                                        dtype,
                                                        num_hidden_layers,
                                                        att_bias)
        torch_model_other = nntile_model.to_torch()
        nntile_model.unregister()

        rtol = dtype2tol[dtype]['rtol']
        for (n1, p1), (n2, p2) in zip(torch_model.named_parameters(),
                torch_model_other.named_parameters()):
            assert n1 == n2
            assert torch.norm(p1 - p2) <= rtol * torch.norm(p1)

    def test_forward(self, starpu_simple, torch_rng,
                     params: LlamaTestParams,
                     dtype: str,
                     num_hidden_layers: int,
                     att_bias: bool):
        torch_model, nntile_model, x, pos_ids, _ = generate_inputs(params,
                dtype, num_hidden_layers, att_bias)
        y = torch_model(x, position_ids=torch.tensor(pos_ids),
                        return_dict=True)
        y_torch = y.logits
        nntile_model.forward_async()
        y_nntile = torch.Tensor(to_numpy(nntile_model.activations[-1].value).T)
        nntile_model.unregister()

        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(y_torch - y_nntile) <= rtol * torch.norm(y_torch)

    def test_forward_backward(self, starpu_simple, torch_rng,
                              params: LlamaTestParams,
                              dtype: str,
                              num_hidden_layers: int,
                              att_bias: bool):
        torch_model, nntile_model, x, pos_ids, y_grad = generate_inputs(params,
                dtype, num_hidden_layers, att_bias)
        y = torch_model(x, position_ids=torch.tensor(pos_ids))
        y_torch = y.logits
        nntile_model.forward_async()
        y_nntile = torch.Tensor(to_numpy(nntile_model.activations[-1].value).T)
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
