# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_llama_attention.py
# Test for nntile.layer.LlamaAttention
# Each test is generated in float precision by Torch, then it is downcasted
# into NNTile type. So, implementation of double precision is NOT checked.
#
# @version 1.1.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from transformers.models.llama.modeling_llama import (
    LlamaAttention as LlamaAttention_torch, LlamaConfig as LlamaConfig_torch)

import nntile
from nntile.model.llama_config import LlamaConfigNNTile
from nntile.tensor import TensorMoments, TensorTraits
from nntile.utils.constructors import to_numpy

# NNTile dtype via corresponding Tensor type
dtype2nntile = {
        'fp32': nntile.tensor.Tensor_fp32,
        'fp32_fast_tf32': nntile.tensor.Tensor_fp32_fast_tf32,
        'bf16': nntile.tensor.Tensor_bf16,
}

dtype2tol = {
        'fp32': {'rtol': 1e-6},
        'fp32_fast_tf32': {'rtol': 8e-4},
        'bf16': {'rtol': 1.6e-2},
}

nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda')


@dataclass
class LlamaAttentionTestParams:
    n_emb: int
    n_emb_tile: int
    n_seq: int
    n_seq_tile: int
    n_batch: int
    n_batch_tile: int
    n_head: int
    n_head_tile: int
    n_head_kv: int
    layer_idx: int = 0
    theta = 2.0


single_tile = LlamaAttentionTestParams(
    n_emb=128,
    n_emb_tile=128,
    n_seq=64,
    n_seq_tile=64,
    n_batch=3,
    n_batch_tile=3,
    n_head=8,
    n_head_tile=4,
    n_head_kv=4,
)

multiple_tiles = LlamaAttentionTestParams(
    n_emb=128,
    n_emb_tile=32,
    n_seq=64,
    n_seq_tile=16,
    n_batch=4,
    n_batch_tile=1,
    n_head=16,
    n_head_tile=8,
    n_head_kv=4,
)


def generate_inputs(dtype: str, params: LlamaAttentionTestParams, bias: bool):
    rng = np.random.default_rng(42)
    torch_layer_config = LlamaConfig_torch(
        hidden_size=params.n_emb,
        num_attention_heads=params.n_head,
        num_key_value_heads=params.n_head_kv,
        attention_bias=bias,
        use_cache=False,
        attention_dropout=0.0,
        rope_theta=params.theta,
    )
    nntile_layer_config = LlamaConfigNNTile(
        hidden_size=params.n_emb,
        hidden_size_tile=params.n_emb_tile,
        n_attention_head=params.n_head,
        num_key_value_heads=params.n_head_kv,
        attention_bias=bias,
        attention_dropout=0.0,
        rope_theta=params.theta,
        n_head_tile=params.n_head_tile,
        num_hidden_layers=torch_layer_config.num_hidden_layers,
        max_position_embeddings=torch_layer_config.max_position_embeddings,
        intermediate_size=torch_layer_config.intermediate_size,
        intermediate_size_tile=torch_layer_config.intermediate_size,
        vocab_size=torch_layer_config.vocab_size,
        vocab_embed_dim_tile=params.n_emb)

    torch_layer = LlamaAttention_torch(
        torch_layer_config, layer_idx=params.layer_idx
    )
    x_shape = [params.n_emb, params.n_seq, params.n_batch]
    x_basetile = [params.n_emb_tile, params.n_seq_tile, params.n_batch_tile]
    x_traits = TensorTraits(x_shape, x_basetile)
    x_distr = [0] * x_traits.grid.nelems
    x_type = dtype2nntile[dtype]
    x_value = x_type(x_traits, x_distr, 0)
    x_grad = x_type(x_traits, x_distr, 0)
    X = TensorMoments(x_value, x_grad, grad_required=True)
    x_random = rng.standard_normal(x_shape)
    x_nntile = np.array(x_random, dtype=np.float32, order="F")
    x_value.from_array(x_nntile)
    x_torch = torch.Tensor(x_nntile.T)

    pos_ids = rng.integers(params.n_seq,
            size=(params.n_batch, params.n_seq),
            dtype=np.int64)
    pos_ids_torch = torch.tensor(pos_ids, dtype=torch.long)
    mask = rng.integers(2, size=(params.n_seq, params.n_seq))
    mask_np = np.array(mask, dtype=bool, order='F')
    mask_torch = torch.Tensor(np.array(1 - mask, dtype=np.float32)).T \
            * torch.finfo(torch.float32).min
    mask_torch = mask_torch[None, None, :, :].expand(params.n_batch, 1, -1, -1)

    nntile_layer, _ = nntile.layer.LlamaAttention.from_torch(
            torch_layer, X, pos_ids, mask_np, nntile_layer_config, 0)
    y_grad_random = rng.standard_normal(x_shape)
    y_grad_nntile = np.array(y_grad_random, dtype=np.float32, order="F")
    nntile_layer.y.grad.from_array(y_grad_nntile)
    y_grad_torch = torch.Tensor(y_grad_nntile.T)
    return torch_layer, nntile_layer, x_torch, pos_ids_torch, mask_torch, \
            y_grad_torch


@pytest.mark.parametrize('bias', [
    False,
    True
    ])
@pytest.mark.parametrize('params', [
    pytest.param(single_tile, id='single_tile'),
    pytest.param(multiple_tiles, id='multiple_tiles'),
])
@pytest.mark.parametrize('dtype', [
    'fp32',
    pytest.param('fp32_fast_tf32', marks=nocuda),
    pytest.param('bf16', marks=nocuda),
])
class TestLlamaAttention:

    def test_torch_coercion(self, starpu_simple, torch_rng, dtype: str,
                            params: LlamaAttentionTestParams, bias: bool):
        torch_layer, nntile_layer, *_ = \
                generate_inputs(dtype, params, bias)
        torch_layer_other = nntile_layer.to_torch()
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()

        rtol = dtype2tol[dtype]['rtol']
        for (n1, p1), (n2, p2) in zip(torch_layer.named_parameters(),
                torch_layer_other.named_parameters()):
            assert n1 == n2
            assert torch.norm(p1 - p2) <= rtol * torch.norm(p1)

    def test_forward(self, starpu_simple, torch_rng, dtype: str,
                     params: LlamaAttentionTestParams, bias: bool):
        torch_layer, nntile_layer, x, pos_ids, mask, *_ = \
                generate_inputs(dtype, params, bias)
        y, _, _ = torch_layer(x, position_ids=pos_ids, attention_mask=mask)
        nntile_layer.forward_async()
        y_nntile = torch.Tensor(to_numpy(nntile_layer.y.value).T)
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()

        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(y - y_nntile) <= rtol * torch.norm(y)

    def test_forward_backward(self, starpu_simple, torch_rng, dtype: str,
                              params: LlamaAttentionTestParams, bias: bool):
        torch_layer, nntile_layer, x, pos_ids, mask, y_grad = \
                generate_inputs(dtype, params, bias)
        y, _, _ = torch_layer(x, position_ids=pos_ids, attention_mask=mask)
        nntile_layer.forward_async()
        y_nntile = torch.Tensor(to_numpy(nntile_layer.y.value).T)
        res = (y * y_grad).sum()
        res.backward()
        nntile_layer.backward_async()
        torch_layer_other = nntile_layer.to_torch_with_grads()
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()

        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(y - y_nntile) <= rtol * torch.norm(y)

        for (n1, p1), (n2, p2) in zip(torch_layer.named_parameters(),
                torch_layer_other.named_parameters()):
            assert n1 == n2
            assert p1.requires_grad == p2.requires_grad
            if p1.requires_grad:
                g1, g2 = p1.grad, p2.grad
                assert torch.norm(g1 - g2) <= rtol * torch.norm(g1)

    def test_flops_counting(self, starpu_simple, torch_rng, dtype: str,
                            params: LlamaAttentionTestParams, bias: bool):

        _, nntile_layer, *_ = \
                generate_inputs(dtype, params, bias)
        analytical_fwd_flops = (4 * params.n_batch * params.n_seq *
                                params.n_emb * (params.n_emb +
                                params.n_emb * params.n_head_kv //
                                params.n_head) + 4 * params.n_batch *
                                params.n_seq**2 * params.n_emb)
        assert (nntile_layer.get_forward_flops() ==
                analytical_fwd_flops)
        assert (nntile_layer.get_backward_flops() ==
                2 * analytical_fwd_flops)
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()
