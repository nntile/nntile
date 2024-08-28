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
from gen_utils import (
    generate_greedy_logits_dynamic_kvcache, generate_greedy_logits_padding)
from transformers.models.llama.modeling_llama import (
    LlamaAttention as LlamaAttention_torch, LlamaConfig as LlamaConfig_torch)

import nntile
import nntile.utils.constructors as nntc
from nntile.model.llama_config import LlamaConfigNNTile
from nntile.tensor import TensorMoments, TensorTraits, clear_async
from nntile.utils.constructors import to_numpy, zeros_like

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


single_tile_trivial = LlamaAttentionTestParams(
    n_emb=4,
    n_emb_tile=4,
    n_seq=10,
    n_seq_tile=10,
    n_batch=1,
    n_batch_tile=1,
    n_head=2,
    n_head_tile=2,
    n_head_kv=2,
)


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


def generate_inputs(dtype: str, params: LlamaAttentionTestParams, bias: bool,
                    flash_attention: bool):
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
        vocab_embed_dim_tile=params.n_emb,
        flash_attention=flash_attention)

    torch_layer = LlamaAttention_torch(
        torch_layer_config, layer_idx=params.layer_idx
    )
    x_shape = [params.n_emb, params.n_seq, params.n_batch]
    x_basetile = [params.n_emb_tile, params.n_seq_tile, params.n_batch_tile]
    x_traits = TensorTraits(x_shape, x_basetile)
    x_distr = [0] * x_traits.grid.nelems
    x_type = dtype2nntile[dtype]
    x_value = x_type(x_traits, x_distr, 0)
    x_grad = zeros_like(x_value)
    X = TensorMoments(x_value, x_grad, grad_required=True)
    x_random = rng.standard_normal(x_shape)
    x_nntile = np.array(x_random, dtype=np.float32, order="F")
    x_value.from_array(x_nntile)
    x_torch = torch.tensor(x_nntile.T, requires_grad=True)

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
    # True # Temporarily disabled to investigate later
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
@pytest.mark.parametrize('flash_attention', [False, True])
class TestLlamaAttention:

    def test_torch_coercion(self, starpu_simple, torch_rng, dtype: str,
                            params: LlamaAttentionTestParams, bias: bool,
                            flash_attention: bool):
        torch_layer, nntile_layer, *_ = \
                generate_inputs(dtype, params, bias, flash_attention)
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
                     params: LlamaAttentionTestParams, bias: bool,
                            flash_attention: bool):
        torch_layer, nntile_layer, x, pos_ids, mask, *_ = \
                generate_inputs(dtype, params, bias, flash_attention)
        y, _, _ = torch_layer(x, position_ids=pos_ids, attention_mask=mask)
        nntile_layer.forward_async()
        y_nntile = torch.Tensor(to_numpy(nntile_layer.y.value).T)
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()

        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(y - y_nntile) <= rtol * torch.norm(y)

    def test_backward(self, starpu_simple, torch_rng, dtype: str,
                              params: LlamaAttentionTestParams, bias: bool,
                            flash_attention: bool):
        torch_layer, nntile_layer, x, pos_ids, mask, y_grad = \
                generate_inputs(dtype, params, bias, flash_attention)
        y, _, _ = torch_layer(x, position_ids=pos_ids, attention_mask=mask)
        nntile_layer.forward_async()
        res = (y * y_grad).sum()
        res.backward()
        for tensor in nntile_layer.parameters:
            if tensor.grad_required:
                clear_async(tensor.grad)
        nntile_layer.backward_async()
        x_grad_nntile = torch.Tensor(to_numpy(nntile_layer.x.grad).T)
        torch_layer_other = nntile_layer.to_torch_with_grads()
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()

        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(x.grad - x_grad_nntile) <= rtol * torch.norm(x.grad)

        for (n1, p1), (n2, p2) in zip(torch_layer.named_parameters(),
                torch_layer_other.named_parameters()):
            assert n1 == n2
            assert p1.requires_grad == p2.requires_grad
            if p1.requires_grad:
                g1, g2 = p1.grad, p2.grad
                assert torch.norm(g1 - g2) <= rtol * torch.norm(g1)

    def test_flops_counting(self, starpu_simple, torch_rng, dtype: str,
                            params: LlamaAttentionTestParams, bias: bool,
                            flash_attention: bool):

        _, nntile_layer, *_ = \
                generate_inputs(dtype, params, bias, flash_attention)
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


@pytest.mark.parametrize("bias", [False])
@pytest.mark.parametrize(
    "params",
    [
        pytest.param(single_tile, id="single_tile"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        "fp32",
    ],
)
@pytest.mark.parametrize("flash_attention", [False])
def test_llama_attn_forward_dynamic(
    starpu_simple,
    torch_rng,
    dtype: str,
    params: LlamaAttentionTestParams,
    bias: bool,
    flash_attention: bool,
):
    torch_layer, nntile_layer, x, pos_ids, mask, *_ = generate_inputs(
        dtype, params, bias, flash_attention
    )

    y, _, _ = torch_layer(x, position_ids=pos_ids, attention_mask=mask)

    input_x = x.cpu().detach().numpy().T
    y_nntile_logits = nntile_layer.forward_dynamic(
        TensorMoments(nntc.from_array(input_x), None, False)
    )
    y_nntile = torch.Tensor(nntc.to_numpy(y_nntile_logits.value).T)

    rtol = dtype2tol[dtype]["rtol"]
    assert torch.norm(y - y_nntile) <= rtol * torch.norm(y)

    truncate_to_size = 4
    y_nntile_logits_trunc = nntile_layer.forward_dynamic(
        TensorMoments(
            nntc.from_array(input_x[:, :truncate_to_size, :]), None, False
        )
    )
    y_nntile_trunc = torch.Tensor(nntc.to_numpy(y_nntile_logits_trunc.value).T)

    y_torch_trunc, _, _ = torch_layer(
        x[:, :truncate_to_size, :],
        position_ids=pos_ids[:, :truncate_to_size],
        attention_mask=mask[:, :, :truncate_to_size, :truncate_to_size],
    )

    rtol = dtype2tol[dtype]["rtol"]
    assert torch.norm(y_torch_trunc - y_nntile_trunc) <= rtol * torch.norm(
        y_torch_trunc
    )

    nntile_layer.unregister()
    nntile_layer.x.unregister()
    nntile_layer.y.unregister()


@pytest.mark.parametrize("bias", [False])
@pytest.mark.parametrize(
    "params",
    [
        pytest.param(single_tile, id="single_tile"),
        pytest.param(single_tile_trivial, id="single_tile_trivial"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        "fp32",
    ],
)
@pytest.mark.parametrize("flash_attention", [False])
def test_llama_attn_kvcache(
    starpu_simple,
    torch_rng,
    dtype: str,
    params: LlamaAttentionTestParams,
    bias: bool,
    flash_attention: bool,
):
    _, nntile_layer, x, _, _, *_ = generate_inputs(
        dtype, params, bias, flash_attention
    )

    prefill_size = 4
    max_tokens = 8

    inp_np = x.cpu().detach().numpy().T
    inp_prefill = nntc.from_array(inp_np[:, :prefill_size, 0:1])

    outs_dyn = generate_greedy_logits_dynamic_kvcache(
        nntile_layer, inp_prefill, prefill_size, max_tokens
    )
    outs_dyn_np = nntc.to_numpy(outs_dyn)

    inp_prefill = nntc.from_array(inp_np[:, :prefill_size, 0:1])
    outs_stat = generate_greedy_logits_padding(
        nntile_layer, inp_prefill, prefill_size, max_tokens
    )
    outs_stat_np = nntc.to_numpy(outs_stat)

    np.testing.assert_allclose(
        outs_stat_np,
        outs_dyn_np,
        err_msg="test_kvcache: Dynamic does not match static",
        rtol=dtype2tol[dtype]["rtol"],
        atol=dtype2tol[dtype]["rtol"],
    )

    nntile_layer.unregister()
    nntile_layer.x.unregister()
    nntile_layer.y.unregister()
