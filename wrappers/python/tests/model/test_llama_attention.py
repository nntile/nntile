# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/model/test_llama_attention.py
# Test for nntile.model.LlamaAttention
# Each test is generated in float precision by PyTorch, then it is downcasted
# into NNTile type. So, implementation of double precision is NOT checked.
#
# @version 1.1.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from transformers.models.llama.modeling_llama import (
    LlamaAttention, LlamaConfig, LlamaRotaryEmbedding)

import nntile
from nntile.model.llama_attention import (
    LlamaAttention as LlamaAttention_nntile)
from nntile.model.llama_config import LlamaConfigNNTile
from nntile.tensor import TensorMoments, TensorTraits
from nntile.utils.constructors import to_numpy

# NNTile dtype via corresponding Tensor type
dtype2nntile = {
    'fp32': nntile.tensor.Tensor_fp32,
    'fp32_fast_tf32': nntile.tensor.Tensor_fp32_fast_tf32,
    'fp32_fast_fp16': nntile.tensor.Tensor_fp32_fast_fp16,
    'fp32_fast_bf16': nntile.tensor.Tensor_fp32_fast_bf16,
    'bf16': nntile.tensor.Tensor_bf16,
    'fp16': nntile.tensor.Tensor_fp16,
}

dtype2tol = {
    'fp32': {'rtol': 1e-5},
    'fp32_fast_tf32': {'rtol': 2e-3},
    'fp32_fast_fp16': {'rtol': 8e-3},
    'fp32_fast_bf16': {'rtol': 8e-3},
    'fp16': {'rtol': 5e-3},
    'bf16': {'rtol': 4e-2},
}

nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda')


@dataclass
class LlamaAttentionTestParams:
    head_size: int
    n_head: int
    n_head_tile: int
    n_head_kv: int
    seq_len: int
    seq_len_tile: int
    n_batch: int
    n_batch_tile: int
    redux: bool = False


single_tile = LlamaAttentionTestParams(
    head_size=64,
    n_head=8,
    n_head_tile=8,
    n_head_kv=4,
    seq_len=64,
    seq_len_tile=64,
    n_batch=3,
    n_batch_tile=3,
)

multiple_tiles = LlamaAttentionTestParams(
    head_size=128,
    n_head=8,
    n_head_tile=2,
    n_head_kv=4,
    seq_len=128,
    seq_len_tile=32,
    n_batch=4,
    n_batch_tile=1,
)


def generate_inputs(params: LlamaAttentionTestParams, dtype: str,
                    flash_attention: bool):
    hidden_size = params.head_size * params.n_head
    torch_layer_config = LlamaConfig(
        hidden_size=hidden_size,
        num_attention_heads=params.n_head,
        num_key_value_heads=params.n_head_kv,
        attention_bias=False,
        pretraining_tp=1,
    )
    torch_layer = LlamaAttention(torch_layer_config, layer_idx=0)

    nntile_config = LlamaConfigNNTile(
        vocab_size=torch_layer_config.vocab_size,
        vocab_embed_dim_tile=hidden_size,
        max_position_embeddings=torch_layer_config.max_position_embeddings,
        n_attention_head=params.n_head,
        n_head_tile=params.n_head_tile,
        num_key_value_heads=params.n_head_kv,
        hidden_size=hidden_size,
        hidden_size_tile=params.head_size,
        intermediate_size=hidden_size * 4,
        intermediate_size_tile=params.head_size * 2,
        dtype=dtype,
        redux=params.redux,
        attention_bias=False,
        flash_attention=flash_attention,
    )

    x_shape = [hidden_size, params.seq_len, params.n_batch]
    x_basetile = [params.head_size,
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
    x_torch = torch.tensor(x_nntile.T, requires_grad=True)

    pos_ids = gen.integers(params.seq_len,
                           size=(params.n_batch, params.seq_len),
                           dtype=np.int64)
    pos_ids_torch = torch.tensor(pos_ids, dtype=torch.long)
    rotary_emb = LlamaRotaryEmbedding(config=torch_layer_config)
    pos_embs_torch = rotary_emb(torch_layer.v_proj.weight, pos_ids_torch)
    mask = np.array(np.triu(np.ones((params.seq_len, params.seq_len))),
                    dtype=bool, order="F")
    nntile_layer = LlamaAttention_nntile.from_torch(torch_layer, X,
                                                    pos_ids, mask,
                                                    nntile_config)
    nntile_layer.clear_gradients()
    y_grad_random = gen.standard_normal(x_shape, dtype=np.float32)
    y_grad_nntile = np.array(y_grad_random, dtype=np.float32, order="F")
    nntile_layer.activations[-1].grad.from_array(y_grad_nntile)
    y_grad_torch = torch.Tensor(y_grad_nntile.T)
    return torch_layer, nntile_layer, x_torch, y_grad_torch, pos_ids_torch, \
        pos_embs_torch, mask


@pytest.mark.parametrize('params', [
    pytest.param(single_tile, id='single_tile'),
    pytest.param(multiple_tiles, id='multiple_tiles'),
])
@pytest.mark.parametrize('dtype', [
    'fp32',
    pytest.param('fp32_fast_tf32', marks=nocuda),
    pytest.param('fp32_fast_fp16', marks=nocuda),
    pytest.param('fp32_fast_bf16', marks=nocuda),
    pytest.param('bf16', marks=nocuda),
    pytest.param('fp16', marks=nocuda),
])
@pytest.mark.parametrize('flash_attention', [False, pytest.param(True, marks=nocuda)])  # noqa: E501
class TestLlamaAttention:

    def test_forward(self, context, torch_rng,
                     params: LlamaAttentionTestParams,
                     dtype: str,
                     flash_attention: bool):
        if flash_attention and dtype not in ('bf16', 'fp16'):
            pytest.skip("Flash SDPA supports only fp16 and bf16")
        torch_layer, nntile_layer, x, _, pos_ids, pos_embs, mask = \
            generate_inputs(params, dtype, flash_attention)
        mask_torch = torch.Tensor(np.array(1 - mask, dtype=np.float32)).T \
            * torch.finfo(torch.float32).min
        mask_torch = mask_torch[None, None, :, :].expand(params.n_batch,
                                                         1, -1, -1)
        y = torch_layer(x, position_embeddings=pos_embs,
                        attention_mask=mask_torch,
                        position_ids=pos_ids)[0]
        nntile_layer.forward_async()
        nntile.starpu.wait_for_all()
        y_nntile = torch.Tensor(
            to_numpy(nntile_layer.activations[-1].value).T
        )
        nntile_layer.unregister()
        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(y - y_nntile) <= rtol * torch.norm(y)

    def test_forward_backward(self, context, torch_rng,
                      params: LlamaAttentionTestParams, dtype: str,
                      flash_attention: bool):
        if flash_attention and dtype not in ('fp16', 'bf16'):
            pytest.skip("Flash SDPA supports only fp16 and bf16")
        torch_layer, nntile_layer, x, y_grad, pos_ids, pos_embs, mask = \
            generate_inputs(params, dtype, flash_attention)
        torch_layer_other = nntile_layer.to_torch()
        mask_torch = torch.Tensor(np.array(1 - mask, dtype=np.float32)).T \
            * torch.finfo(torch.float32).min
        mask_torch = mask_torch[None, None, :, :].expand(params.n_batch,
                                                         1, -1, -1)
        y = torch_layer(x, position_embeddings=pos_embs,
                        attention_mask=mask_torch,
                        position_ids=pos_ids)[0]
        nntile_layer.forward_async()
        res = (y * y_grad).sum()
        res.backward()
        nntile_layer.backward_async()
        x_grad_nntile = torch.Tensor(
                to_numpy(nntile_layer.activations[0].grad).T)
        torch_layer_other = nntile_layer.to_torch_with_grads()
        nntile_layer.unregister()

        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(x.grad - x_grad_nntile) <= rtol * torch.norm(x.grad)

        for (n1, p1), (n2, p2) in zip(torch_layer.named_parameters(),
                torch_layer_other.named_parameters()):
            assert n1 == n2
            assert p1.requires_grad == p2.requires_grad
            if p1.requires_grad:
                g1, g2 = p1.grad, p2.grad
                assert torch.norm(g1 - g2) <= rtol * torch.norm(g1)
