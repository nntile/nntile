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
# @version 1.1.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from gen_utils import (
    generate_greedy_logits_dynamic_kvcache, generate_greedy_logits_padding)
from transformers.models.gpt_neo.modeling_gpt_neo import (
    GPTNeoAttention as GPTNeoAttentionTorch, GPTNeoConfig as GPTNeoConfigTorch)

import nntile
from nntile.model.gpt_neo_config import GPTNeoConfig
from nntile.tensor import TensorMoments, TensorTraits
from nntile.utils.constructors import to_numpy

# NNTile dtype via corresponding Tensor type
dtype2nntile = {
        'fp32': nntile.tensor.Tensor_fp32,
        'bf16': nntile.tensor.Tensor_bf16,
        'fp32_fast_tf32': nntile.tensor.Tensor_fp32_fast_tf32,
        'fp32_fast_fp16': nntile.tensor.Tensor_fp32_fast_fp16,
        'fp32_fast_bf16': nntile.tensor.Tensor_fp32_fast_bf16,
}

dtype2np = {
    'fp16': np.float16,
    'bf16': np.float16,
    'fp32': np.float32,
}

dtype2tol = {
        'fp32': {'rtol': 1e-6},
        'bf16': {'rtol': 4e-2},
        'fp32_fast_tf32': {'rtol': 2e-3},
        'fp32_fast_fp16': {'rtol': 8e-3},
        'fp32_fast_bf16': {'rtol': 8e-3},
}

nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda')


@dataclass
class GPTNeoAttentionTestParams:
    n_emb: int
    n_emb_tile: int
    n_seq: int
    n_seq_tile: int
    n_batch: int
    n_batch_tile: int
    n_head: int
    n_head_tile: int
    layer_id: int = 0
    window_size: int = 256


single_tile = GPTNeoAttentionTestParams(
    n_emb=128,
    n_emb_tile=128,
    n_seq=64,
    n_seq_tile=64,
    n_batch=3,
    n_batch_tile=3,
    n_head=4,
    n_head_tile=4,
    layer_id=0
)

multiple_tiles = GPTNeoAttentionTestParams(
    n_emb=128,
    n_emb_tile=32,
    n_seq=64,
    n_seq_tile=16,
    n_batch=4,
    n_batch_tile=1,
    n_head=16,
    n_head_tile=8,
    layer_id=1,
)

multiple_tiles_small_window = GPTNeoAttentionTestParams(
    n_emb=128,
    n_emb_tile=32,
    n_seq=64,
    n_seq_tile=16,
    n_batch=4,
    n_batch_tile=1,
    n_head=16,
    n_head_tile=8,
    layer_id=1,
    window_size=16,
)


def generate_inputs(dtype: str, params: GPTNeoAttentionTestParams):
    rng = np.random.default_rng(42)
    torch_layer_config = GPTNeoConfigTorch(
        hidden_size=params.n_emb,
        num_heads=params.n_head,
        use_cache=False,
        resid_dropout=0.0,
        attention_dropout=0.0,
        window_size=params.window_size
    )

    nntile_config = GPTNeoConfig(
        vocab_size=torch_layer_config.vocab_size,
        vocab_embed_dim_tile=params.n_emb,
        hidden_size=params.n_emb,
        hidden_size_tile=params.n_emb_tile,
        intermediate_size=torch_layer_config.intermediate_size,
        intermediate_size_tile=torch_layer_config.intermediate_size,
        num_heads=params.n_head,
        num_heads_tile=params.n_head_tile,
        attention_types=torch_layer_config.attention_types,
        dtype=dtype,
        window_size=params.window_size
    )

    torch_layer = GPTNeoAttentionTorch(
        torch_layer_config,
        layer_id=params.layer_id
    )

    x_shape = [params.n_emb, params.n_seq, params.n_batch]
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
    nntile_layer = nntile.layer.GPTNeoAttention.from_torch(
            torch_layer, X, X, X, nntile_config
    )
    nntile_layer.clear_gradients()

    y_grad_random = rng.standard_normal(nntile_layer.y.grad.shape)
    y_grad_nntile = np.array(y_grad_random, dtype=np.float32, order="F")
    nntile_layer.y.grad.from_array(y_grad_nntile)
    y_grad_torch = torch.Tensor(y_grad_nntile.T)
    return torch_layer, nntile_layer, x_torch, y_grad_torch


@pytest.mark.parametrize('params', [
    pytest.param(single_tile, id='single_tile'),
    pytest.param(multiple_tiles, id='multiple_tiles'),
    pytest.param(multiple_tiles, id='multiple_tiles_small_window'),
])
@pytest.mark.parametrize('dtype', [
    'fp32',
    pytest.param('bf16', marks=nocuda),
    pytest.param('fp32_fast_tf32', marks=nocuda),
    pytest.param('fp32_fast_fp16', marks=nocuda),
    pytest.param('fp32_fast_bf16', marks=nocuda),
])
class TestGPTNeoAttention:

    def test_torch_coercion(self, context, torch_rng, dtype: str,
                            params: GPTNeoAttentionTestParams):
        torch_layer, nntile_layer, *_ = generate_inputs(dtype, params)
        torch_layer_other = nntile_layer.to_torch()
        nntile_layer.unregister()
        nntile_layer.x_q.unregister()
        nntile_layer.x_k.unregister()
        nntile_layer.x_v.unregister()
        nntile_layer.y.unregister()

        rtol = dtype2tol[dtype]['rtol']
        for (n1, p1), (n2, p2) in zip(torch_layer.named_parameters(),
                torch_layer_other.named_parameters()):
            assert n1 == n2
            assert torch.norm(p1 - p2) <= rtol * torch.norm(p1)

    def test_forward(self, context, torch_rng, dtype: str,
                     params: GPTNeoAttentionTestParams):
        torch_layer, nntile_layer, x, _ = generate_inputs(dtype, params)
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

    def test_backward(self, context, torch_rng, dtype: str,
                              params: GPTNeoAttentionTestParams):
        torch_layer, nntile_layer, x, y_grad = generate_inputs(dtype, params)
        y, _ = torch_layer(x)
        nntile_layer.forward_async()
        res = (y * y_grad).sum()
        res.backward()
        nntile_layer.backward_async()
        torch_layer_other = nntile_layer.to_torch_with_grads()
        grad_nntile = torch.Tensor(to_numpy(nntile_layer.x_k.grad).T)
        nntile_layer.unregister()
        nntile_layer.x_q.unregister()
        nntile_layer.x_k.unregister()
        nntile_layer.x_v.unregister()
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
    "n_head,n_head_tile,n_emb,n_emb_tile,seq_size", [(2, 1, 6, 2, 10)]
)
def test_dynamic(
    context, numpy_rng, n_head, n_head_tile, n_emb, n_emb_tile, seq_size
):
    input_shape = (n_emb, seq_size, 1)
    inp_np = np.asfortranarray(numpy_rng.random(input_shape))

    inp = nntile.utils.constructors.from_array(
        inp_np, basetile_shape=(n_emb_tile,) + input_shape[1:]
    )
    inp2 = nntile.utils.constructors.from_array(
        inp_np, basetile_shape=(n_emb_tile,) + input_shape[1:]
    )
    inp3 = nntile.utils.constructors.from_array(
        inp_np, basetile_shape=(n_emb_tile,) + input_shape[1:]
    )

    inp_tm = nntile.tensor.TensorMoments(
        inp,
        grad=nntile.utils.constructors.zeros(inp.shape, dtype=type(inp)),
        grad_required=False
    )
    inp_tm2 = nntile.tensor.TensorMoments(
        inp2,
        grad=nntile.utils.constructors.zeros(inp2.shape, dtype=type(inp)),
        grad_required=False
    )
    inp_tm3 = nntile.tensor.TensorMoments(
        inp3,
        grad=nntile.utils.constructors.zeros(inp3.shape, dtype=type(inp)),
        grad_required=False
    )

    attn = nntile.layer.GPTNeoAttention.generate_simple(
        inp_tm,
        inp_tm2,
        inp_tm3,
        n_head,
        n_head_tile,
        layer_id=0,
        attention_type="global"
    )
    attn.init_randn_async()

    attn.forward_async()
    out_dynamic_expected_np = nntile.utils.constructors.to_numpy(
        attn.y.value)

    out_dynamic_actual, _ = attn.forward_dynamic(inp_tm)
    out_dynamic_actual_np = nntile.utils.constructors.to_numpy(
        out_dynamic_actual.value)

    np.testing.assert_allclose(
        out_dynamic_actual_np,
        out_dynamic_expected_np,
        err_msg="Dynamic does not match static",
    )


@pytest.mark.parametrize("n_head,n_head_tile", [(1, 1)])
def test_kvcache(context, numpy_rng, n_head, n_head_tile):
    prefill_size = 4
    max_tokens = 8

    inp_np = np.asfortranarray(numpy_rng.random((3, 8, 1)))
    inp_np[:, prefill_size:, :] = 0

    inp = nntile.utils.constructors.from_array(inp_np)
    inp2 = nntile.utils.constructors.from_array(inp_np)
    inp3 = nntile.utils.constructors.from_array(inp_np)

    inp_tm = nntile.tensor.TensorMoments(
        inp,
        grad=nntile.utils.constructors.zeros(inp.shape, dtype=type(inp)),
        grad_required=False
    )
    inp_tm2 = nntile.tensor.TensorMoments(
        inp2,
        grad=nntile.utils.constructors.zeros(inp2.shape, dtype=type(inp)),
        grad_required=False
    )
    inp_tm3 = nntile.tensor.TensorMoments(
        inp3,
        grad=nntile.utils.constructors.zeros(inp3.shape, dtype=type(inp)),
        grad_required=False
    )

    attn = nntile.layer.GPTNeoAttention.generate_simple(
        inp_tm,
        inp_tm2,
        inp_tm3,
        n_head,
        n_head_tile,
        layer_id=0,
        attention_type="global"
    )
    attn.init_randn_async()

    # slice to prefill size
    inp_prefill = nntile.utils.constructors.from_array(
        inp_np[:, :prefill_size, :])
    outs_dyn = generate_greedy_logits_dynamic_kvcache(
        attn, inp_prefill, prefill_size, max_tokens
    )
    outs_dyn_np = nntile.utils.constructors.to_numpy(outs_dyn)

    inp_prefill = nntile.utils.constructors.from_array(
        inp_np[:, :prefill_size, :])
    outs_stat = generate_greedy_logits_padding(
        attn, inp_prefill, prefill_size, max_tokens
    )
    outs_stat_np = nntile.utils.constructors.to_numpy(outs_stat)

    np.testing.assert_allclose(
        outs_stat_np,
        outs_dyn_np,
        err_msg="test_kvcache: Dynamic does not match static",
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['bf16', 'fp32'])
def test_bench_gpt_neo_attention_forward_async(context_cuda, benchmark_operation, dtype: str):
    params = single_tile
    _, nntile_layer, *_ = generate_inputs(dtype, params)

    np_dtype = dtype2np[dtype]
    out_np = np.zeros(nntile_layer.y.value.shape, dtype=np_dtype, order='F')

    def bench_fn():
        nntile_layer.forward_async()
        nntile_layer.y.value.to_array(out_np)

    nntile.starpu.wait_for_all()
    benchmark_operation(bench_fn)


@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['bf16', 'fp32'])
def test_bench_gpt_neo_attention_backward_async(context_cuda, benchmark_operation, dtype: str):
    params = single_tile
    _, nntile_layer, *_ = generate_inputs(dtype, params)

    nntile_layer.clear_gradients()
    rng = np.random.default_rng(42)
    np_dtype = dtype2np[dtype]
    grad_np = np.array(rng.standard_normal(nntile_layer.y.value.shape), dtype=np_dtype, order='F')

    def bench_fn():
        nntile_layer.forward_async()
        nntile_layer.y.grad.from_array(grad_np)
        nntile_layer.backward_async()

    nntile.starpu.wait_for_all()
    benchmark_operation(bench_fn)
