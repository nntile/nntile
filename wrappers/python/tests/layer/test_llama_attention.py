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
# @version 1.0.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from transformers.models.llama.modeling_llama import (
    LlamaAttention as LlamaAttention_torch, LlamaConfig as LlamaConfig_torch)

import nntile
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
        'fp32_fast_tf32': {'rtol': 6e-4},
        'bf16': {'rtol': 1.6e-2},
}

nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda')


def assert_close_by_frobnorm(a: np.ndarray, b: np.ndarray, rtol: float):
    np.testing.assert_array_less(
            np.linalg.norm(a - b),
            rtol * np.linalg.norm(a)
    )


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

    pos_ids = rng.integers(
            params.n_seq,
            size=(params.n_batch, params.n_seq),
            dtype=np.int64
    )
    pos_ids_torch = torch.tensor(pos_ids, dtype=torch.long)
    mask = rng.integers(2, size=(params.n_seq, params.n_seq))
    mask_np = np.array(mask, dtype=bool, order='F')
    mask_torch = torch.Tensor(np.array(1 - mask, dtype=np.float32)).T \
            * torch.finfo(torch.float32).min
    mask_torch = mask_torch[None, None, :, :].expand(params.n_batch, 1, -1, -1)

    nntile_layer = nntile.layer.LlamaAttention.from_torch(
            torch_layer, X, params.n_head_tile, pos_ids, mask_np, params.theta
    )
    y_grad_random = rng.standard_normal(x_shape)
    y_grad_nntile = np.array(y_grad_random, dtype=np.float32, order="F")
    nntile_layer.y.grad.from_array(y_grad_nntile)
    y_grad_torch = torch.Tensor(y_grad_nntile.T)
    return torch_layer, nntile_layer, x_torch, pos_ids_torch, mask_torch, \
            y_grad_torch


@pytest.mark.parametrize('bias', [False, True])
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
        assert_close_by_frobnorm(
            torch_layer.q_proj.weight.detach().numpy(),
            torch_layer_other.q_proj.weight.detach().numpy(),
            **dtype2tol[dtype]
        )
        assert_close_by_frobnorm(
            torch_layer.k_proj.weight.detach().numpy(),
            torch_layer_other.k_proj.weight.detach().numpy(),
            **dtype2tol[dtype]
        )
        assert_close_by_frobnorm(
            torch_layer.v_proj.weight.detach().numpy(),
            torch_layer_other.v_proj.weight.detach().numpy(),
            **dtype2tol[dtype]
        )
        assert_close_by_frobnorm(
            torch_layer.o_proj.weight.detach().numpy(),
            torch_layer_other.o_proj.weight.detach().numpy(),
            **dtype2tol[dtype]
        )
        if bias:
            assert_close_by_frobnorm(
                torch_layer.q_proj.bias.detach().numpy(),
                torch_layer_other.q_proj.bias.detach().numpy(),
                **dtype2tol[dtype]
            )
            assert_close_by_frobnorm(
                torch_layer.k_proj.bias.detach().numpy(),
                torch_layer_other.k_proj.bias.detach().numpy(),
                **dtype2tol[dtype]
            )
            assert_close_by_frobnorm(
                torch_layer.v_proj.bias.detach().numpy(),
                torch_layer_other.v_proj.bias.detach().numpy(),
                **dtype2tol[dtype]
            )
            assert_close_by_frobnorm(
                torch_layer.o_proj.bias.detach().numpy(),
                torch_layer_other.o_proj.bias.detach().numpy(),
                **dtype2tol[dtype]
            )

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
        assert_close_by_frobnorm(
                y.detach().numpy(),
                y_nntile.detach().numpy(),
                **dtype2tol[dtype]
        )

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
        assert_close_by_frobnorm(
                y.detach().numpy(),
                y_nntile.detach().numpy(),
                **dtype2tol[dtype]
        )
        assert_close_by_frobnorm(
            torch_layer.q_proj.weight.grad.detach().numpy(),
            torch_layer_other.q_proj.weight.grad.detach().numpy(),
            **dtype2tol[dtype]
        )
        assert_close_by_frobnorm(
            torch_layer.k_proj.weight.grad.detach().numpy(),
            torch_layer_other.k_proj.weight.grad.detach().numpy(),
            **dtype2tol[dtype]
        )
        assert_close_by_frobnorm(
            torch_layer.v_proj.weight.grad.detach().numpy(),
            torch_layer_other.v_proj.weight.grad.detach().numpy(),
            **dtype2tol[dtype]
        )
        assert_close_by_frobnorm(
            torch_layer.o_proj.weight.grad.detach().numpy(),
            torch_layer_other.o_proj.weight.grad.detach().numpy(),
            **dtype2tol[dtype]
        )
        if bias:
            assert_close_by_frobnorm(
                torch_layer.q_proj.bias.grad.detach().numpy(),
                torch_layer_other.q_proj.bias.grad.detach().numpy(),
                **dtype2tol[dtype]
            )
            assert_close_by_frobnorm(
                torch_layer.k_proj.bias.grad.detach().numpy(),
                torch_layer_other.k_proj.bias.grad.detach().numpy(),
                **dtype2tol[dtype]
            )
            assert_close_by_frobnorm(
                torch_layer.v_proj.bias.grad.detach().numpy(),
                torch_layer_other.v_proj.bias.grad.detach().numpy(),
                **dtype2tol[dtype]
            )
            assert_close_by_frobnorm(
                torch_layer.o_proj.bias.grad.detach().numpy(),
                torch_layer_other.o_proj.bias.grad.detach().numpy(),
                **dtype2tol[dtype]
            )
