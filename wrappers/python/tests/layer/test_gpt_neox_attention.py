# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_gpt_neox_attention.py
# Test for nntile.layer.GPTNeoXAttention
#
# @version 1.1.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention as AttentionTorch, GPTNeoXConfig as ConfigTorch,
    GPTNeoXRotaryEmbedding as RotaryEmbeddingTorch)

import nntile
from nntile.model.gpt_neox_config import GPTNeoXConfig
from nntile.tensor import TensorMoments, TensorTraits, clear_async
from nntile.utils.constructors import to_numpy
from gen_utils import (
    generate_greedy_logits_dynamic_kvcache, generate_greedy_logits_padding
)

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
class GPTNeoXAttentionTestParams:
    n_emb: int
    n_emb_tile: int
    n_seq: int
    n_seq_tile: int
    n_batch: int
    n_batch_tile: int
    n_head: int
    n_head_tile: int
    layer_idx: int = 0


single_tile_trivial = GPTNeoXAttentionTestParams(
    n_emb=4,
    n_emb_tile=4,
    n_seq=10,
    n_seq_tile=10,
    n_batch=1,
    n_batch_tile=1,
    n_head=2,
    n_head_tile=2,
)


single_tile = GPTNeoXAttentionTestParams(
    n_emb=128,
    n_emb_tile=128,
    n_seq=64,
    n_seq_tile=64,
    n_batch=4,
    n_batch_tile=4,
    n_head=8,
    n_head_tile=8,
)

multiple_tiles = GPTNeoXAttentionTestParams(
    n_emb=128,
    n_emb_tile=32,
    n_seq=64,
    n_seq_tile=16,
    n_batch=4,
    n_batch_tile=1,
    n_head=16,
    n_head_tile=8,
)


def generate_inputs(dtype: str, params: GPTNeoXAttentionTestParams):
    rng = np.random.default_rng(42)
    torch_layer_config = ConfigTorch(
        hidden_size=params.n_emb,
        num_attention_heads=params.n_head,
        rotary_pct=1.0,
        use_cache=False,
        attention_bias=False,
        attention_dropout=0.0,
    )

    nntile_config = GPTNeoXConfig(
        vocab_size=torch_layer_config.vocab_size,
        vocab_embed_dim_tile=params.n_emb,
        hidden_size=params.n_emb,
        hidden_size_tile=params.n_emb_tile,
        intermediate_size=torch_layer_config.intermediate_size,
        intermediate_size_tile=torch_layer_config.intermediate_size,
        num_heads=params.n_head,
        num_heads_tile=params.n_head_tile,
        dtype=dtype,
        redux=False,
    )

    torch_layer = AttentionTorch(
        torch_layer_config
    )
    n_emb, n_seq, n_batch = params.n_emb, params.n_seq, params.n_batch
    x_shape = [n_emb, n_seq, n_batch]
    x_basetile = [params.n_emb_tile, params.n_seq_tile, params.n_batch_tile]
    x_type = dtype2nntile[dtype]

    x_q_traits = TensorTraits(x_shape, x_basetile)
    x_q_distr = [0] * x_q_traits.grid.nelems
    x_value = x_type(x_q_traits, x_q_distr)
    x_grad = x_type(x_q_traits, x_q_distr)
    X = TensorMoments(x_value, x_grad, grad_required=True)

    x_random = rng.standard_normal(x_shape)
    x_nntile = np.array(x_random, dtype=np.float32, order="F")
    x_value.from_array(x_nntile)
    x_torch = torch.Tensor(x_nntile.T)
    x_torch.requires_grad_()

    pos_ids = rng.integers(n_seq, size=(n_batch, n_seq), dtype=np.int64)
    pos_ids_torch = torch.tensor(pos_ids, dtype=torch.long)

    rotary_emb = RotaryEmbeddingTorch(config=torch_layer_config)
    pos_embs_torch = rotary_emb(
        torch_layer.query_key_value.weight[2 * n_emb: 3 * n_emb, :],
        pos_ids_torch
    )

    mask_np = np.array(
            np.triu(np.ones((n_seq, n_seq))), dtype=bool, order="F"
        )
    mask_torch = torch.Tensor(np.array(1 - mask_np, dtype=np.float32)).T \
            * torch.finfo(torch.float32).min
    mask_torch = mask_torch[None, None, :, :].expand(n_batch, 1, -1, -1)

    nntile_layer = nntile.layer.GPTNeoXAttention.from_torch(
        torch_layer, X, pos_ids, mask_np, nntile_config
    )

    y_grad_random = rng.standard_normal(nntile_layer.y.grad.shape)
    y_grad_nntile = np.array(y_grad_random, dtype=np.float32, order="F")
    nntile_layer.y.grad.from_array(y_grad_nntile)
    y_grad_torch = torch.Tensor(y_grad_nntile.T)
    return torch_layer, nntile_layer, x_torch, pos_embs_torch, \
            mask_torch, y_grad_torch


@pytest.mark.parametrize('params', [
    pytest.param(single_tile, id='single_tile'),
    pytest.param(single_tile_trivial, id='single_tile_trivial'),
    pytest.param(multiple_tiles, id='multiple_tiles'),
])
@pytest.mark.parametrize('dtype', [
    'fp32',
    pytest.param('fp32_fast_tf32', marks=nocuda),
    pytest.param('bf16', marks=nocuda),
])
class TestGPTNeoXAttention:

    def test_torch_coercion(self, context, torch_rng, dtype: str,
                            params: GPTNeoXAttentionTestParams):
        torch_layer, nntile_layer, *_ = generate_inputs(dtype, params)
        torch_layer_other = nntile_layer.to_torch()
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()

        rtol = dtype2tol[dtype]['rtol']
        for (n1, p1), (n2, p2) in zip(torch_layer.named_parameters(),
                torch_layer_other.named_parameters()):
            assert n1 == n2
            assert torch.norm(p1 - p2) <= rtol * torch.norm(p1)

    def test_forward(self, context, torch_rng, dtype: str,
                     params: GPTNeoXAttentionTestParams):
        torch_layer, nntile_layer, x, pos_embs, mask, *_ = \
            generate_inputs(dtype, params)
        y, _ = torch_layer(
            x, attention_mask=mask, position_embeddings=pos_embs
        )
        nntile_layer.forward_async()
        y_nntile = torch.Tensor(to_numpy(nntile_layer.y.value).T)
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()
        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(y - y_nntile) <= rtol * torch.norm(y)

    def test_backward(self, context, torch_rng, dtype: str,
                      params: GPTNeoXAttentionTestParams):
        torch_layer, nntile_layer, x, pos_embs, mask, y_grad = \
            generate_inputs(dtype, params)
        y, _ = torch_layer(
            x, attention_mask=mask, position_embeddings=pos_embs
        )
        nntile_layer.forward_async()
        res = (y * y_grad).sum()
        res.backward()
        for tensor in nntile_layer.parameters:
            if tensor.grad_required:
                clear_async(tensor.grad)
        nntile_layer.backward_async()
        torch_layer_other = nntile_layer.to_torch_with_grads()
        grad_nntile = torch.Tensor(to_numpy(nntile_layer.x.grad).T)
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()

        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(x.grad - grad_nntile) <= rtol * torch.norm(x.grad)

        for (n1, p1), (n2, p2) in zip(torch_layer.named_parameters(),
                torch_layer_other.named_parameters()):
            assert n1 == n2
            assert p1.requires_grad == p2.requires_grad
            if p1.requires_grad:
                g1, g2 = p1.grad, p2.grad
                assert torch.norm(g1 - g2) <= rtol * torch.norm(g1)


@pytest.mark.parametrize(
    "n_head,n_head_tile,n_emb,n_emb_tile,seq_size", [(2, 1, 8, 2, 10)]
)
def test_dynamic(
    context, numpy_rng, n_head, n_head_tile, n_emb, n_emb_tile, seq_size
):
    input_shape = (n_emb, seq_size, 1)
    inp_np = np.asfortranarray(numpy_rng.random(input_shape))

    inp = nntile.utils.constructors.from_array(
        inp_np, basetile_shape=(n_emb_tile,) + input_shape[1:]
    )

    inp_tm = nntile.tensor.TensorMoments(
        inp, grad=nntile.utils.constructors.zeros(inp.shape, dtype=type(inp)), grad_required=False
    )

    # Create position_ids for the GPTNeoXAttention layer
    position_ids = np.arange(seq_size, dtype=np.int64)[None, :]  # (1, seq_size)

    # Create causal mask
    causal_mask = np.array(
        np.triu(np.ones((seq_size, seq_size))), dtype=bool, order="F"
    )

    attn = nntile.layer.GPTNeoXAttention.generate_simple(
        inp_tm, n_head, n_head_tile, position_ids, theta=10000.0,
        bias=False, mask=causal_mask, redux=False
    )
    attn.init_randn_async()
    
    attn.forward_async()
    out_dynamic_expected_np = nntile.utils.constructors.to_numpy(attn.y.value)

    out_dynamic_actual, _ = attn.forward_dynamic(inp_tm)
    out_dynamic_actual_np = nntile.utils.constructors.to_numpy(out_dynamic_actual.value)

    np.testing.assert_allclose(
        out_dynamic_actual_np,
        out_dynamic_expected_np,
        err_msg="Dynamic does not match static",
    )


@pytest.mark.parametrize("n_head,n_head_tile", [(1, 1)])
def test_kvcache(context, numpy_rng, n_head, n_head_tile):
    prefill_size = 4
    max_tokens = 8

    inp_np = np.asfortranarray(numpy_rng.random((4, 8, 1)))
    inp_np[:, prefill_size:, :] = 0

    inp = nntile.utils.constructors.from_array(inp_np)

    inp_tm = nntile.tensor.TensorMoments(
        inp, grad=nntile.utils.constructors.zeros(inp.shape, dtype=type(inp)), grad_required=False
    )

    # Create position_ids for the GPTNeoXAttention layer
    position_ids = np.arange(8, dtype=np.int64)[None, :]  # (1, 8)

    # Create causal mask
    causal_mask = np.array(
        np.triu(np.ones((8, 8))), dtype=bool, order="F"
    )

    attn = nntile.layer.GPTNeoXAttention.generate_simple(
        inp_tm, n_head, n_head_tile, position_ids, theta=10000.0,
        bias=False, mask=causal_mask, redux=False
    )
    attn.init_randn_async()

    # slice to prefill size
    inp_prefill = nntile.utils.constructors.from_array(inp_np[:, :prefill_size, :])
    outs_dyn = generate_greedy_logits_dynamic_kvcache(
        attn, inp_prefill, prefill_size, max_tokens
    )
    outs_dyn_np = nntile.utils.constructors.to_numpy(outs_dyn)

    inp_prefill = nntile.utils.constructors.from_array(inp_np[:, :prefill_size, :])
    outs_stat = generate_greedy_logits_padding(
        attn, inp_prefill, prefill_size, max_tokens
    )
    outs_stat_np = nntile.utils.constructors.to_numpy(outs_stat)

    np.testing.assert_allclose(
        outs_stat_np,
        outs_dyn_np,
        err_msg="test_kvcache: Dynamic does not match static",
    )
