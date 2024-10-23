# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/model/test_bert_attention.py
# Test for nntile.layer.BertAttention
#
# @version 1.1.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from transformers.models.bert.modeling_bert import (
    BertAttention as BertAttention_torch, BertConfig as BertConfig_torch)

import nntile
from nntile.model.bert_config import BertConfigNNTile
from nntile.model.bert_modules import BertAttention as BertAttentionNNTile
from nntile.tensor import TensorMoments, TensorTraits
from nntile.utils.constructors import to_numpy

# NNTile dtype via corresponding Tensor type
dtype2nntile = {
        'fp32': nntile.tensor.Tensor_fp32,
        'fp32_fast_tf32': nntile.tensor.Tensor_fp32_fast_tf32,
        'fp32_fast_bf16': nntile.tensor.Tensor_fp32_fast_bf16,
        'fp32_fast_fp16': nntile.tensor.Tensor_fp32_fast_fp16,
        'bf16': nntile.tensor.Tensor_bf16,

}

dtype2tol = {
        'fp32': {'rtol': 1e-6},
        'fp32_fast_tf32': {'rtol': 7e-4},
        'fp32_fast_bf16': {'rtol': 1.6e-2},
        'fp32_fast_fp16': {'rtol': 7e-4},
        'bf16': {'rtol': 1.6e-2},
}

nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda')


@dataclass
class BertAttentionTestParams:
    n_emb: int
    n_emb_tile: int
    n_seq: int
    n_seq_tile: int
    n_batch: int
    n_batch_tile: int
    n_head: int
    n_head_tile: int
    layer_idx: int = 0


single_tile = BertAttentionTestParams(
    n_emb=128,
    n_emb_tile=128,
    n_seq=64,
    n_seq_tile=64,
    n_batch=3,
    n_batch_tile=3,
    n_head=4,
    n_head_tile=4,
)

multiple_tiles = BertAttentionTestParams(
    n_emb=128,
    n_emb_tile=32,
    n_seq=64,
    n_seq_tile=16,
    n_batch=4,
    n_batch_tile=1,
    n_head=16,
    n_head_tile=8,
)


def generate_inputs(dtype: str, params: BertAttentionTestParams):
    rng = np.random.default_rng(42)
    torch_layer_config = BertConfig_torch(
        hidden_size=params.n_emb,
        num_attention_heads=params.n_head,
        use_cache=False,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        _attn_implementation="eager",
        add_cross_attention=False,
    )

    nntile_config = BertConfigNNTile(
        vocab_size=torch_layer_config.vocab_size,
        vocab_embed_dim_tile=params.n_emb,
        hidden_size=params.n_emb,
        hidden_size_tile=params.n_emb_tile,
        intermediate_size=torch_layer_config.intermediate_size,
        intermediate_size_tile=torch_layer_config.intermediate_size,
        num_attention_heads=params.n_head,
        n_head_tile=params.n_head_tile,
        dtype=dtype,
    )

    torch_layer = BertAttention_torch(
        torch_layer_config
    )

    x_shape = [params.n_emb, params.n_seq, params.n_batch]
    x_basetile = [params.n_emb_tile, params.n_seq_tile, params.n_batch_tile]
    x_type = dtype2nntile[dtype]

    x_q_traits = TensorTraits(x_shape, x_basetile)
    x_q_distr = [0] * x_q_traits.grid.nelems
    x_value = x_type(x_q_traits, x_q_distr, 0)
    x_grad = x_type(x_q_traits, x_q_distr, 0)
    X = TensorMoments(x_value, x_grad, grad_required=True)

    x_random = rng.standard_normal(x_shape)
    x_nntile = np.array(x_random, dtype=np.float32, order="F")
    x_value.from_array(x_nntile)
    x_torch = torch.Tensor(x_nntile.T)
    x_torch.requires_grad_(True)
    nntile_layer, _ = BertAttentionNNTile.from_torch(
            torch_layer, X, nntile_config, 0
    )
    nntile_layer.clear_gradients()
    y_grad_random = rng.standard_normal((params.n_emb,
                                         params.n_seq,
                                         params.n_batch),
                                         dtype=np.float32)
    y_grad_nntile = np.array(y_grad_random, dtype=np.float32, order="F")
    nntile_layer.activations[-1].grad.from_array(y_grad_nntile)
    y_grad_torch = torch.Tensor(y_grad_nntile.T)
    return torch_layer, nntile_layer, x_torch, y_grad_torch


@pytest.mark.parametrize('params', [
    pytest.param(single_tile, id='single_tile'),
    pytest.param(multiple_tiles, id='multiple_tiles'),
])
@pytest.mark.parametrize('dtype', [
    'fp32',
    pytest.param('fp32_fast_tf32', marks=nocuda),
    pytest.param('bf16', marks=nocuda),
    pytest.param('fp32_fast_bf16', marks=nocuda),
    pytest.param('fp32_fast_fp16', marks=nocuda),
])
class TestBertAttention:

    def test_torch_coercion(self, starpu_simple, torch_rng, dtype: str,
                            params: BertAttentionTestParams):
        torch_layer, nntile_layer, *_ = generate_inputs(dtype, params)
        torch_layer_other = nntile_layer.to_torch()
        nntile_layer.unregister()

        rtol = dtype2tol[dtype]['rtol']
        for (n1, p1), (n2, p2) in zip(torch_layer.named_parameters(),
                torch_layer_other.named_parameters()):
            assert n1 == n2
            assert torch.norm(p1 - p2) <= rtol * torch.norm(p1)

    def test_forward(self, starpu_simple, torch_rng, dtype: str,
                     params: BertAttentionTestParams):
        torch_layer, nntile_layer, x, _ = generate_inputs(dtype, params)
        y = torch_layer(x)[0]
        nntile_layer.forward_async()
        y_nntile = torch.Tensor(to_numpy(nntile_layer.activations[-1].value).T)
        nntile_layer.unregister()
        rtol = dtype2tol[dtype]['rtol']
        print(torch.norm(y_nntile), torch.norm(y))
        print(y_nntile.shape, y.shape)
        assert torch.norm(y - y_nntile) <= \
            rtol * torch.norm(y)

    def test_backward(self, starpu_simple, torch_rng, dtype: str,
                              params: BertAttentionTestParams):
        torch_layer, nntile_layer, x, y_grad = generate_inputs(dtype, params)
        y = torch_layer(x)[0]
        nntile_layer.forward_async()
        res = (y * y_grad).sum()
        res.backward()
        nntile_layer.backward_async()
        torch_layer_other = nntile_layer.to_torch_with_grads()
        grad_nntile = torch.Tensor(
            to_numpy(nntile_layer.activations[0].grad).T)
        nntile_layer.unregister()

        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(x.grad - grad_nntile) <= rtol * torch.norm(x.grad)
        for (n1, p1), (n2, p2) in zip(torch_layer.named_parameters(),
                torch_layer_other.named_parameters()):
            assert n1 == n2
            assert p1.requires_grad == p2.requires_grad
            # Bias gradients in selfattention are unstable,
            # so we concatenate them over q,k,v and test together below
            if n1.split(".")[2] == "bias" and n1.split(".")[0] == "self":
                continue
            if p1.requires_grad:
                g1, g2 = p1.grad, p2.grad
                assert torch.norm(g1 - g2) <= rtol * torch.norm(g1)
        bias_grad_torch = torch.hstack([torch_layer.self.query.bias.grad,
                                        torch_layer.self.key.bias.grad,
                                        torch_layer.self.value.bias.grad])
        bias_grad_nntile = torch.hstack(
            [torch_layer_other.self.query.bias.grad,
             torch_layer_other.self.key.bias.grad,
             torch_layer_other.self.value.bias.grad])
        assert torch.norm(bias_grad_torch - bias_grad_nntile) <= \
            rtol * torch.norm(bias_grad_torch)
