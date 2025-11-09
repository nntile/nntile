# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/model/test_gpt_neox_model.py
# Test for nntile.model.GPTNeoXModel
#
# @version 1.1.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXConfig as ConfigTorch, GPTNeoXModel as ModelTorch)

import nntile
import nntile.utils.constructors as nntc
from nntile.model.gpt_neox_config import GPTNeoXConfig
from nntile.model.gpt_neox_model import GPTNeoXModel
from nntile.utils.constructors import to_numpy

# NNTile dtype via corresponding Tensor type
dtype2nntile = {
        'fp32': nntile.tensor.Tensor_fp32,
        'fp32_fast_tf32': nntile.tensor.Tensor_fp32_fast_tf32,
        'bf16': nntile.tensor.Tensor_bf16,
}

dtype2np = {
        'fp32': np.float32,
        'bf16': np.float16,
        'fp16': np.float16,
}

dtype2tol = {
        'fp32': {'rtol': 1e-5},
        'fp32_fast_tf32': {'rtol': 3e-3},
        'bf16': {'rtol': 4e-2},
}

nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda')


@dataclass
class GPTNeoXModelTestParams:
    vocab_size: int
    vocab_embed_dim_tile: int
    n_emb: int
    n_emb_tile: int
    n_seq: int
    n_seq_tile: int
    intermediate_size: int
    intermediate_size_tile: int
    n_batch: int
    n_batch_tile: int
    n_head: int
    n_head_tile: int
    max_position_embeddings: int
    layer_norm_epsilon: float


single_tile_trivial = GPTNeoXModelTestParams(
    vocab_size=1024,
    vocab_embed_dim_tile=128,
    n_emb=16,
    n_emb_tile=16,
    n_seq=10,
    n_seq_tile=10,
    intermediate_size=16,
    intermediate_size_tile=16,
    n_batch=1,
    n_batch_tile=1,
    n_head=2,
    n_head_tile=2,
    max_position_embeddings=2048,
    layer_norm_epsilon=1e-5,
)


single_tile = GPTNeoXModelTestParams(
    vocab_size=32000,
    vocab_embed_dim_tile=128,
    n_emb=128,
    n_emb_tile=128,
    n_seq=64,
    n_seq_tile=64,
    intermediate_size=64,
    intermediate_size_tile=64,
    n_batch=4,
    n_batch_tile=4,
    n_head=8,
    n_head_tile=8,
    max_position_embeddings=2048,
    layer_norm_epsilon=1e-5,
)

multiple_tiles = GPTNeoXModelTestParams(
    vocab_size=32000,
    vocab_embed_dim_tile=32,
    n_emb=128,
    n_emb_tile=32,
    n_seq=64,
    n_seq_tile=16,
    intermediate_size=64,
    intermediate_size_tile=16,
    n_batch=4,
    n_batch_tile=1,
    n_head=16,
    n_head_tile=8,
    max_position_embeddings=2048,
    layer_norm_epsilon=1e-5,
)


def generate_inputs(params: GPTNeoXModelTestParams,
                    dtype: str,
                    num_hidden_layers: int,
                    rotary_pct: float,
                    use_parallel_residual: bool,
                    att_bias: bool):
    rng = np.random.default_rng(42)
    torch_model_config = ConfigTorch(
        vocab_size=params.vocab_size,
        hidden_size=params.n_emb,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=params.n_head,
        intermediate_size=params.intermediate_size,
        rotary_pct=rotary_pct,
        attention_bias=att_bias,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        max_position_embeddings=params.max_position_embeddings,
        layer_norm_eps=params.layer_norm_epsilon,
        use_cache=False,
        use_parallel_residual=use_parallel_residual,
    )

    nntile_config = GPTNeoXConfig(
        vocab_size=params.vocab_size,
        vocab_embed_dim_tile=params.vocab_embed_dim_tile,
        hidden_size=params.n_emb,
        hidden_size_tile=params.n_emb_tile,
        intermediate_size=params.intermediate_size,
        intermediate_size_tile=params.intermediate_size_tile,
        num_heads=params.n_head,
        num_heads_tile=params.n_head_tile,
        dtype=dtype,
        layer_norm_epsilon=params.layer_norm_epsilon,
        max_position_embeddings=params.max_position_embeddings,
        num_hidden_layers=num_hidden_layers,
        redux=False,
        rotary_pct=rotary_pct,
        use_parallel_residual=use_parallel_residual,
        attention_bias=att_bias,
    )
    torch_model = ModelTorch(
        torch_model_config,
    )
    n_seq, n_batch = params.n_seq, params.n_batch

    pos_ids = rng.integers(params.n_seq,
                           size=(params.n_batch, params.n_seq),
                           dtype=np.int64)

    mask_np = np.array(
            np.triu(np.ones((n_seq, n_seq))), dtype=bool, order="F"
        )
    mask_torch = torch.Tensor(np.array(1 - mask_np, dtype=np.float32)).T \
            * torch.finfo(torch.float32).min
    mask_torch = mask_torch[None, None, :, :].expand(n_batch, 1, -1, -1)

    nntile_model = GPTNeoXModel.from_torch(
            torch_model, params.n_batch, params.n_batch_tile,
            params.n_seq, params.n_seq_tile, pos_ids,
            mask_np, nntile_config)

    nntile_model.clear_gradients()
    x_random = rng.integers(params.vocab_size,
                            size=nntile_model.activations[0].value.shape,
                            dtype=np.int64)

    x_nntile = np.array(x_random, dtype=np.int64, order='F')
    nntile_model.activations[0].value.from_array(x_nntile)
    x_torch = torch.tensor(x_nntile.T)
    y_grad_random = rng.standard_normal((params.n_emb,
                                         params.n_seq,
                                         params.n_batch),
                                        dtype=np.float32)
    y_grad_nntile = np.array(y_grad_random, dtype=np.float32, order='F')
    nntile_model.activations[-1].grad.from_array(y_grad_nntile)
    y_grad_torch = torch.Tensor(y_grad_nntile.T)
    return torch_model, nntile_model, x_torch, pos_ids, mask_torch, \
            y_grad_torch


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
@pytest.mark.parametrize('num_hidden_layers', [0, 1, 2])
@pytest.mark.parametrize('att_bias', [
    False, True,
])
@pytest.mark.parametrize('rotary_pct', [
    0.25, 1.0,
])
@pytest.mark.parametrize('use_parallel_residual', [False, True])
class TestGPTNeoXModel:
    def test_torch_coercion(self, context, torch_rng,
                      params: GPTNeoXModelTestParams,
                      dtype: str,
                      num_hidden_layers: int,
                      rotary_pct: float,
                      use_parallel_residual: bool,
                      att_bias: bool):
        torch_model, nntile_model, *_ = generate_inputs(
            params, dtype, num_hidden_layers, rotary_pct,
            use_parallel_residual, att_bias
        )
        torch_model_other = nntile_model.to_torch()
        nntile_model.unregister()

        rtol = dtype2tol[dtype]['rtol']
        for (n1, p1), (n2, p2) in zip(torch_model.named_parameters(),
                torch_model_other.named_parameters()):
            assert n1 == n2
            assert torch.norm(p1 - p2) <= rtol * torch.norm(p1)

    def test_forward(self, context, torch_rng,
                     params: GPTNeoXModelTestParams,
                     dtype: str,
                     num_hidden_layers: int,
                     rotary_pct: float,
                     use_parallel_residual: bool,
                     att_bias: bool):
        torch_model, nntile_model, x, pos_ids, mask, _ = \
            generate_inputs(
                params, dtype, num_hidden_layers, rotary_pct,
                use_parallel_residual, att_bias
            )
        y = torch_model(x,
                        attention_mask=mask,
                        position_ids=torch.tensor(pos_ids),
                        return_dict=True)
        y_torch = y.last_hidden_state
        nntile_model.forward_async()
        y_nntile = torch.Tensor(
            to_numpy(nntile_model.activations[-1].value).T
        )
        nntile_model.unregister()

        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(y_torch - y_nntile) <= rtol * torch.norm(y_torch)

    def test_forward_backward(self, context, torch_rng,
                              params: GPTNeoXModelTestParams,
                              dtype: str,
                              num_hidden_layers: int,
                              rotary_pct: float,
                              use_parallel_residual: bool,
                              att_bias: bool):
        torch_model, nntile_model, x, pos_ids, mask, y_grad = \
            generate_inputs(params, dtype, num_hidden_layers, rotary_pct,
                            use_parallel_residual, att_bias
            )
        y = torch_model(x,
                        attention_mask=mask,
                        position_ids=torch.tensor(pos_ids),
                        return_dict=True)
        y_torch = y.last_hidden_state
        nntile_model.forward_async()
        y_nntile = torch.Tensor(
            to_numpy(nntile_model.activations[-1].value).T
        )
        res = (y.last_hidden_state * y_grad).sum()
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


@pytest.mark.parametrize('params', [
    pytest.param(single_tile, id='single_tile'),
])
@pytest.mark.parametrize('dtype', [
    'fp32',
])
@pytest.mark.parametrize('num_hidden_layers', [2])
@pytest.mark.parametrize('rotary_pct', [0.25])
@pytest.mark.parametrize('att_bias', [False, True])
@pytest.mark.parametrize('use_parallel_residual', [False, True])
def test_forward_dynamic(context, torch_rng,
                     params: GPTNeoXModelTestParams,
                     dtype: str,
                     num_hidden_layers: int,
                     rotary_pct: float,
                     use_parallel_residual: bool,
                     att_bias: bool):
    torch_model, nntile_model, x, pos_ids, mask, _ = \
            generate_inputs(
                params, dtype, num_hidden_layers, rotary_pct,
                use_parallel_residual, att_bias
            )
    y = torch_model(x,
                    attention_mask=mask,
                    position_ids=torch.tensor(pos_ids),
                    return_dict=True)
    y_torch = y.last_hidden_state

    x_np = x.cpu().detach().numpy()
    x_nnt = nntile.tensor.TensorMoments(nntc.from_array(x_np.T), None, False)

    logits_nnt, _ = nntile_model.forward_dynamic(x_nnt)
    y_nntile = torch.Tensor(nntc.to_numpy(logits_nnt.value).T)

    rtol = dtype2tol[dtype]['rtol']
    assert torch.norm(y_torch - y_nntile) <= rtol * torch.norm(y_torch)

    x_np_trunc = x_np[:x_np.shape[0] // 2, :x_np.shape[1] // 2]
    pos_ids_trunc = pos_ids[:pos_ids.shape[0] // 2, :pos_ids.shape[1] // 2]
    trunc_n_seq = pos_ids.shape[1] // 2
    trunc_n_batch = pos_ids.shape[0] // 2
    mask_np = np.array(
            np.triu(np.ones((trunc_n_seq, trunc_n_seq))), dtype=bool, order="F"
        )
    mask_trunc = torch.Tensor(np.array(1 - mask_np, dtype=np.float32)).T \
            * torch.finfo(torch.float32).min
    mask_trunc = mask_trunc[None, None, :, :].expand(trunc_n_batch, 1, -1, -1)
    x_trunc_nnt = nntile.tensor.TensorMoments(
        nntc.from_array(x_np_trunc.T), None, False
    )

    y_trunc = torch_model(torch.tensor(x_np_trunc),
                    attention_mask=mask_trunc,
                    position_ids=torch.tensor(pos_ids_trunc),
                    return_dict=True)
    y_trunc_torch = y_trunc.last_hidden_state

    logits_trunc_nnt, _ = nntile_model.forward_dynamic(x_trunc_nnt)
    y_trunc_nntile = torch.Tensor(nntc.to_numpy(logits_trunc_nnt.value).T)

    actual_diff = torch.norm(y_trunc_torch - y_trunc_nntile)
    upper_bound_diff = rtol * torch.norm(y_trunc_torch)
    assert actual_diff <= upper_bound_diff

    nntile_model.unregister()


@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['fp32', 'bf16'])
def test_bench_gptneox_forward_async(context_cuda, benchmark_model, dtype: str):
    params = single_tile_trivial
    num_hidden_layers = 1
    rotary_pct = 0.25
    use_parallel_residual = False
    att_bias = False
    _, nntile_model, *_ = generate_inputs(
        params, dtype, num_hidden_layers, rotary_pct, use_parallel_residual, att_bias
    )

    np_out = np.zeros(
        nntile_model.activations[-1].value.shape, dtype=dtype2np[dtype], order="F"
    )

    def bench_fn():
        nntile_model.forward_async()
        nntile.starpu.wait_for_all()

    nntile.starpu.wait_for_all()
    benchmark_model(bench_fn)
    nntile_model.unregister()


@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['fp32', 'bf16'])
def test_bench_gptneox_forward_backward_async(context_cuda, benchmark_model, dtype: str):
    params = single_tile_trivial
    num_hidden_layers = 1
    rotary_pct = 0.25
    use_parallel_residual = False
    att_bias = False
    _, nntile_model, *_ = generate_inputs(
        params, dtype, num_hidden_layers, rotary_pct, use_parallel_residual, att_bias
    )
    nntile_model.clear_gradients()

    rng = np.random.default_rng(42)
    np_grad = np.array(
        rng.standard_normal(nntile_model.activations[-1].value.shape),
        dtype=dtype2np[dtype],
        order="F",
    )

    def bench_fn():
        nntile_model.clear_gradients()
        nntile_model.forward_async()
        nntile_model.activations[-1].grad.from_array(np_grad)
        nntile_model.backward_async()
        nntile.starpu.wait_for_all()

    nntile.starpu.wait_for_all()
    benchmark_model(bench_fn)
    nntile_model.unregister()
