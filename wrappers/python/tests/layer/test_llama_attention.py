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
    dtype: np.dtype
    bias: bool
    layer_idx: int = 0


TEST_PARAMS = [
    LlamaAttentionTestParams(
        n_emb=128,
        n_emb_tile=32,
        n_seq=64,
        n_seq_tile=16,
        n_batch=4,
        n_batch_tile=1,
        n_head=16,
        n_head_tile=8,
        n_head_kv=4,
        dtype=np.dtypes.Float32DType(),
        bias=True,
    ),
    LlamaAttentionTestParams(
        n_emb=128,
        n_emb_tile=32,
        n_seq=64,
        n_seq_tile=16,
        n_batch=4,
        n_batch_tile=1,
        n_head=16,
        n_head_tile=8,
        n_head_kv=4,
        dtype=np.dtypes.Float32DType(),
        bias=False,
    ),
    LlamaAttentionTestParams(
        n_emb=128,
        n_emb_tile=128,
        n_seq=64,
        n_seq_tile=64,
        n_batch=3,
        n_batch_tile=3,
        n_head=8,
        n_head_tile=4,
        n_head_kv=4,
        dtype=np.dtypes.Float32DType(),
        bias=True,
    ),
    LlamaAttentionTestParams(
        n_emb=128,
        n_emb_tile=128,
        n_seq=64,
        n_seq_tile=64,
        n_batch=3,
        n_batch_tile=3,
        n_head=8,
        n_head_tile=4,
        n_head_kv=4,
        dtype=np.dtypes.Float32DType(),
        bias=False,
    ),
]


def generate_inputs(params: LlamaAttentionTestParams):
    torch_layer_config = LlamaConfig_torch(
        hidden_size=params.n_emb,
        num_attention_heads=params.n_head,
        num_key_value_heads=params.n_head_kv,
        attention_bias=params.bias,
        use_cache=False,
        attention_dropout=0.0,
    )
    torch_layer = LlamaAttention_torch(
        torch_layer_config, layer_idx=params.layer_idx
    )
    x_shape = [params.n_emb, params.n_seq, params.n_batch]
    x_basetile = [params.n_emb_tile, params.n_seq_tile, params.n_batch_tile]
    x_traits = TensorTraits(x_shape, x_basetile)
    x_distr = [0] * x_traits.grid.nelems
    x_type = nntile.utils.constructors.np2nnt_type_mapping[type(params.dtype)]
    x_value = x_type(x_traits, x_distr, 0)
    x_grad = x_type(x_traits, x_distr, 0)
    X = TensorMoments(x_value, x_grad, grad_required=True)
    x_random = np.random.randn(*x_shape)
    x_nntile = np.array(x_random, dtype=params.dtype, order="F")
    x_value.from_array(x_nntile)
    x_torch = torch.Tensor(x_nntile.T)
    nntile_layer = nntile.layer.LlamaAttention.from_torch(
        torch_layer, X, params.n_head_tile
    )
    y_grad_random = np.random.randn(*x_shape)
    y_grad_nntile = np.array(y_grad_random, dtype=params.dtype, order="F")
    nntile_layer.y.grad.from_array(y_grad_nntile)
    y_grad_torch = torch.Tensor(y_grad_nntile.T)
    return torch_layer, nntile_layer, x_torch, y_grad_torch


@pytest.mark.parametrize("params", TEST_PARAMS)
class TestLlamaAttention:
    def test_from_to_torch(
        self, starpu_simple, torch_rng, params: LlamaAttentionTestParams
    ):
        torch_layer, nntile_layer, _, _ = generate_inputs(params)
        torch_layer_other = nntile_layer.to_torch()
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()
        assert torch.equal(
            torch_layer.q_proj.weight, torch_layer_other.q_proj.weight
        )
        assert torch.equal(
            torch_layer.k_proj.weight, torch_layer_other.k_proj.weight
        )
        assert torch.equal(
            torch_layer.v_proj.weight, torch_layer_other.v_proj.weight
        )
        assert torch.equal(
            torch_layer.o_proj.weight, torch_layer_other.o_proj.weight
        )
        if params.bias:
            assert torch.equal(
                torch_layer.q_proj.bias, torch_layer_other.q_proj.bias
            )
            assert torch.equal(
                torch_layer.k_proj.bias, torch_layer_other.k_proj.bias
            )
            assert torch.equal(
                torch_layer.v_proj.bias, torch_layer_other.v_proj.bias
            )
            assert torch.equal(
                torch_layer.o_proj.bias, torch_layer_other.o_proj.bias
            )

    def test_forward(
        self, starpu_simple, torch_rng, params: LlamaAttentionTestParams
    ):
        torch_layer, nntile_layer, x, _ = generate_inputs(params)
        pos_ids = torch.zeros((params.n_batch, params.n_seq), dtype=torch.long)
        y, _, _ = torch_layer(x, position_ids=pos_ids)
        nntile_layer.forward_async()
        y_nntile = torch.Tensor(to_numpy(nntile_layer.y.value).T)
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()
        torch.testing.assert_close(y, y_nntile)

    def test_forward_backward(
        self, starpu_simple, torch_rng, params: LlamaAttentionTestParams
    ):
        torch_layer, nntile_layer, x, y_grad = generate_inputs(params)
        torch_layer_other = nntile_layer.to_torch()
        pos_ids = torch.zeros((params.n_batch, params.n_seq), dtype=torch.long)
        y, _, _ = torch_layer(x, position_ids=pos_ids)
        nntile_layer.forward_async()
        y_nntile = torch.Tensor(to_numpy(nntile_layer.y.value).T)
        res = (y * y_grad).sum()
        res.backward()
        nntile_layer.backward_async()
        torch_layer_other = nntile_layer.to_torch_with_grads()
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()
        torch.testing.assert_close(y, y_nntile)
        torch.testing.assert_close(
            torch_layer.q_proj.weight, torch_layer_other.q_proj.weight
        )
        torch.testing.assert_close(
            torch_layer.k_proj.weight, torch_layer_other.k_proj.weight
        )
        torch.testing.assert_close(
            torch_layer.v_proj.weight, torch_layer_other.v_proj.weight
        )
        torch.testing.assert_close(
            torch_layer.o_proj.weight, torch_layer_other.o_proj.weight
        )
        if params.bias:
            torch.testing.assert_close(
                torch_layer.q_proj.bias, torch_layer_other.q_proj.bias
            )
            torch.testing.assert_close(
                torch_layer.k_proj.bias, torch_layer_other.k_proj.bias
            )
            torch.testing.assert_close(
                torch_layer.v_proj.bias, torch_layer_other.v_proj.bias
            )
            torch.testing.assert_close(
                torch_layer.o_proj.bias, torch_layer_other.o_proj.bias
            )
