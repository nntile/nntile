# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_gpt2_attention.py
# Test for nntile.layer.GPT2Attention
#
# @version 1.0.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention as GPT2Attention_torch, GPT2Config as GPT2Config_torch)

import nntile
from nntile.tensor import TensorMoments, TensorTraits, Tensor_bool
from nntile.utils.constructors import to_numpy

# NNTile dtype via corresponding Tensor type
dtype2nntile = {
        'fp32': nntile.tensor.Tensor_fp32,
        'fp32_fast_tf32': nntile.tensor.Tensor_fp32_fast_tf32,
        'bf16': nntile.tensor.Tensor_bf16,
}

dtype2tol = {
        'fp32': {'rtol': 1e-6},
        'fp32_fast_tf32': {'rtol': 6e-4},
        'bf16': {'rtol': 1.6e-2},
}

nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda')


@dataclass
class GPT2AttentionTestParams:
    n_emb: int
    n_emb_tile: int
    n_seq: int
    n_seq_tile: int
    n_batch: int
    n_batch_tile: int
    n_head: int
    n_head_tile: int
    layer_idx: int = 0


single_tile = GPT2AttentionTestParams(
    n_emb=128,
    n_emb_tile=128,
    n_seq=64,
    n_seq_tile=64,
    n_batch=3,
    n_batch_tile=3,
    n_head=4,
    n_head_tile=4,
)

multiple_tiles = GPT2AttentionTestParams(
    n_emb=128,
    n_emb_tile=32,
    n_seq=64,
    n_seq_tile=16,
    n_batch=4,
    n_batch_tile=1,
    n_head=16,
    n_head_tile=8,
)


def generate_inputs(dtype: str, params: GPT2AttentionTestParams):
    rng = np.random.default_rng(42)
    torch_layer_config = GPT2Config_torch(
        n_embd=params.n_emb,
        n_head=params.n_head,
        use_cache=False,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop = 0.0,
        reorder_and_upcast_attn = False,
        scale_attn_by_inverse_layer_idx = False,
        scale_attn_weights = True
    )
    torch_layer = GPT2Attention_torch(
        torch_layer_config, is_cross_attention=False, layer_idx=params.layer_idx
    )

    x_shape = [params.n_emb, params.n_seq, params.n_batch]
    x_basetile = [params.n_emb_tile, params.n_seq_tile, params.n_batch_tile]
    x_type = dtype2nntile[dtype]

    x_q_traits = TensorTraits(x_shape, x_basetile)
    x_q_distr = [0] * x_q_traits.grid.nelems
    x_value = x_type(x_q_traits, x_q_distr, 0)
    x_grad = x_type(x_q_traits, x_q_distr, 0)
    X = TensorMoments(x_value, x_grad, grad_required=True)

    # x_k_traits = TensorTraits(x_shape, x_basetile)
    # x_k_distr = [0] * x_k_traits.grid.nelems
    # x_k_value = x_type(x_k_traits, x_k_distr, 0)
    # x_k_grad = x_type(x_k_traits, x_k_distr, 0)
    # x_k = TensorMoments(x_k_value, x_k_grad, grad_required=True)

    # x_v_traits = TensorTraits(x_shape, x_basetile)
    # x_v_distr = [0] * x_v_traits.grid.nelems
    # x_v_value = x_type(x_v_traits, x_v_distr, 0)
    # x_v_grad = x_type(x_v_traits, x_v_distr, 0)
    # x_v = TensorMoments(x_v_value, x_v_grad, grad_required=True)
    # X = TensorMoments(x_value, x_grad, grad_required=True)

    x_random = rng.standard_normal(x_shape)
    x_nntile = np.array(x_random, dtype=np.float32, order="F")
    x_value.from_array(x_nntile)
    x_torch = torch.Tensor(x_nntile.T)
   
    nntile_layer = nntile.layer.GPT2Attention.from_torch(
            torch_layer, X, X, X, params.n_head_tile
    )
    return torch_layer, nntile_layer, x_torch


@pytest.mark.parametrize('params', [
    pytest.param(single_tile, id='single_tile'),
    pytest.param(multiple_tiles, id='multiple_tiles'),
])
@pytest.mark.parametrize('dtype', [
    'fp32',
    # pytest.param('fp32_fast_tf32', marks=nocuda),
    # pytest.param('bf16', marks=nocuda),
])

class TestGPT2Attention:

    def test_forward(self, starpu_simple, torch_rng, dtype: str,
                     params: GPT2AttentionTestParams):
        torch_layer, nntile_layer, x = generate_inputs(dtype, params)
        y, _ = torch_layer(x)
        nntile_layer.forward_async()
        y_nntile = torch.Tensor(to_numpy(nntile_layer.y.value).T)
        nntile_layer.unregister()
        nntile_layer.x_q.unregister()
        nntile_layer.x_k.unregister()
        nntile_layer.x_v.unregister()
        nntile_layer.y.unregister()
        print(torch.norm(y_nntile), torch.norm(y))
        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(y - y_nntile) <= rtol * torch.norm(y)
