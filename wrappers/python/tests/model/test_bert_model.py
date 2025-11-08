# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/model/test_bert_model.py
# Test for nntile.model.bert.bert_model
# Each test is generated in float precision by Torch, then it is downcasted
# into NNTile type. So, implementation of double precision is NOT checked.
#
# @version 1.1.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from transformers import BertConfig as BertConfigTorch
from transformers.models.bert.modeling_bert import BertModel as BertModel_torch

import nntile
from nntile.model.bert import BertModel as BertModelNNTile
from nntile.model.bert_config import BertConfigNNTile
from nntile.tensor import to_numpy

# NNTile dtype via corresponding Tensor type
dtype2nntile = {
        'fp32': nntile.tensor.Tensor_fp32,
        'bf16': nntile.tensor.Tensor_bf16,
        'fp32_fast_tf32': nntile.tensor.Tensor_fp32_fast_tf32,
        'fp32_fast_fp16': nntile.tensor.Tensor_fp32_fast_fp16,
        'fp32_fast_bf16': nntile.tensor.Tensor_fp32_fast_bf16,
}

dtype2np = {
        'fp32': np.float32,
        'bf16': np.float16,
        'fp16': np.float16,
}

dtype2tol = {
        'fp32': {'rtol': 1e-5},
        'bf16': {'rtol': 4e-2},
        'fp32_fast_tf32': {'rtol': 2e-3},
        'fp32_fast_fp16': {'rtol': 8e-3},
        'fp32_fast_bf16': {'rtol': 8e-3},
}

nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda')


@dataclass
class BertTestParams:
    vocab_size: int
    vocab_embed_dim_tile: int
    hidden_size: int
    hidden_size_tile: int
    intermediate_size: int
    intermediate_size_tile: int
    batch_size: int
    batch_size_tile: int
    seq_len: int
    seq_len_tile: int
    n_head: int
    n_head_tile: int
    type_vocab_size: int
    redux: bool = True


single_tile = BertTestParams(
    vocab_size=32000,
    vocab_embed_dim_tile=128,
    hidden_size=128,
    hidden_size_tile=128,
    intermediate_size=64,
    intermediate_size_tile=64,
    batch_size=3,
    batch_size_tile=3,
    seq_len=32,
    seq_len_tile=32,
    n_head=16,
    n_head_tile=16,
    type_vocab_size=2
    )

multiple_tiles = BertTestParams(
    vocab_size=32000,
    vocab_embed_dim_tile=32,
    hidden_size=128,
    hidden_size_tile=32,
    intermediate_size=64,
    intermediate_size_tile=16,
    batch_size=3,
    batch_size_tile=1,
    seq_len=128,
    seq_len_tile=32,
    n_head=16,
    n_head_tile=8,
    type_vocab_size=2)


def generate_inputs(params: BertTestParams,
                    dtype: str, num_hidden_layers: int):
    torch_config = BertConfigTorch(
        vocab_size=params.vocab_size,
        hidden_size=params.hidden_size,
        num_attention_heads=params.n_head,
        intermediate_size=params.intermediate_size,
        use_cache=False,
        add_cross_attention=False,
        type_vocab_size=params.type_vocab_size,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0.0,
        _attn_implementation="eager",
        chunk_size_feed_forward=0,
        is_decoder=False,
        num_hidden_layers=num_hidden_layers,
        # TODO: we have to introduce 'padding_idx' parameter in our
        # implementation of Embedding layer
        pad_token_id=None
    )

    torch_model = BertModel_torch(torch_config)
    nntile_config = BertConfigNNTile(
            vocab_size=params.vocab_size,
            vocab_embed_dim_tile=params.vocab_embed_dim_tile,
            hidden_size=params.hidden_size,
            hidden_size_tile=params.hidden_size_tile,
            intermediate_size=params.intermediate_size,
            intermediate_size_tile=params.intermediate_size_tile,
            num_attention_heads=params.n_head,
            n_head_tile=params.n_head_tile,
            dtype=dtype,
            type_vocab_size=params.type_vocab_size,
            activation_function=torch_config.hidden_act,
            num_hidden_layers=num_hidden_layers
    )
    gen = np.random.default_rng(42)

    nntile_model = BertModelNNTile.from_torch(
            torch_model, params.batch_size, params.batch_size_tile,
            params.seq_len, params.seq_len_tile, nntile_config)
    nntile_model.clear_gradients()
    x_random = gen.integers(params.vocab_size,
                            size=nntile_model.activations[0].value.shape,
                            dtype=np.int64)

    x_nntile = np.array(x_random, dtype=np.int64, order='F')
    nntile_model.activations[0].value.from_array(x_nntile)
    x_torch = torch.tensor(x_nntile.T)
    y_grad_random = gen.standard_normal((params.hidden_size,
                                         params.seq_len,
                                         params.batch_size),
                                         dtype=np.float32)
    y_grad_nntile = np.array(y_grad_random, dtype=np.float32, order='F')
    nntile_model.activations[-1].grad.from_array(y_grad_nntile)
    y_grad_torch = torch.Tensor(y_grad_nntile.T)
    return torch_model, nntile_model, x_torch, y_grad_torch


@pytest.mark.parametrize('params', [
    pytest.param(single_tile, id='single_tile'),
    pytest.param(multiple_tiles, id='multiple_tiles'),
])
@pytest.mark.parametrize('dtype', [
    'fp32',
    pytest.param('bf16', marks=nocuda),
    pytest.param('fp32_fast_tf32', marks=nocuda),
    pytest.param('fp32_fast_fp16', marks=nocuda),
    pytest.param('fp32_fast_bf16', marks=nocuda),
])
@pytest.mark.parametrize('num_hidden_layers', [
    pytest.param(1, id='single layer'),
    pytest.param(2, id='two layers'),
    pytest.param(5, id='five layers'),
])
class TestBertModel:
    def test_coercion(self, context, torch_rng,
                      params: BertTestParams,
                      dtype: str, num_hidden_layers: int):

        torch_model, nntile_model, _, _ = generate_inputs(params,
                                                          dtype,
                                                          num_hidden_layers)

        torch_model_other = nntile_model.to_torch()
        nntile_model.unregister()
        rtol = dtype2tol[dtype]['rtol']
        for (n1, p1), (n2, p2) in zip(torch_model.named_parameters(),
                torch_model_other.named_parameters()):
            assert n1 == n2
            assert torch.norm(p1 - p2) <= rtol * torch.norm(p1)

    def test_forward(self, context, torch_rng,
                     params: BertTestParams,
                     dtype: str, num_hidden_layers: int):
        torch_model, nntile_model, x, _ = generate_inputs(params,
                                                          dtype,
                                                          num_hidden_layers)
        y_torch = torch_model(x)[0]
        nntile_model.forward_async()
        y_nntile = torch.Tensor(to_numpy(nntile_model.activations[-1].value).T)
        nntile_model.unregister()

        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(y_torch - y_nntile) <= rtol * torch.norm(y_torch)

    def test_backward(self, context, torch_rng,
                      params: BertTestParams,
                      dtype: str, num_hidden_layers: int):
        torch_model, nntile_model, x, y_grad = generate_inputs(params,
                                                               dtype,
                                                               num_hidden_layers)
        y = torch_model(x)[0]
        nntile_model.forward_async()
        res = (y * y_grad).sum()
        res.backward()

        nntile_model.backward_async()
        torch_model_other = nntile_model.to_torch_with_grads()
        nntile_model.unregister()

        rtol = dtype2tol[dtype]['rtol']
        for (n1, p1), (n2, p2) in zip(torch_model.named_parameters(),
                torch_model_other.named_parameters()):
            assert n1 == n2
            assert p1.requires_grad == p2.requires_grad
            # Bias gradients in selfattention are unstable,
            # so we concatenate them over q,k,v and test together below
            splitted_name = n1.split(".")
            print(n1)
            if (len(splitted_name) == 7 and
                splitted_name[6] == "bias" and
                splitted_name[3] == "attention" and
                splitted_name[4] == "self"):
                continue
            if p1.requires_grad:
                g1, g2 = p1.grad, p2.grad
                assert torch.norm(g1 - g2) <= rtol * torch.norm(g1)
        for layer, layer_other in zip(torch_model.encoder.layer,
                                      torch_model_other.encoder.layer):
            bias_grad_torch = torch.hstack([
                layer.attention.self.query.bias.grad,
                layer.attention.self.key.bias.grad,
                layer.attention.self.value.bias.grad])
            bias_grad_nntile = torch.hstack(
                [layer_other.attention.self.query.bias.grad,
                layer_other.attention.self.key.bias.grad,
                layer_other.attention.self.value.bias.grad])
            assert torch.norm(bias_grad_torch - bias_grad_nntile) <= \
                rtol * torch.norm(bias_grad_torch)


@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['fp32', 'bf16'])
def test_bench_bert_forward_async(context_cuda, benchmark_model, dtype: str):
    params = single_tile
    num_hidden_layers = 1
    _, nntile_model, _, _ = generate_inputs(params, dtype, num_hidden_layers)

    np_out = np.zeros(
        nntile_model.activations[-1].value.shape, dtype=dtype2np[dtype], order="F"
    )

    def bench_fn():
        nntile_model.forward_async()
        nntile_model.activations[-1].value.to_array(np_out)

    nntile.starpu.wait_for_all()
    benchmark_model(bench_fn)
    nntile_model.unregister()


@pytest.mark.benchmark
@pytest.mark.parametrize('dtype', ['fp32', 'bf16'])
def test_bench_bert_backward_async(context_cuda, benchmark_model, dtype: str):
    params = single_tile
    num_hidden_layers = 1
    _, nntile_model, _, _ = generate_inputs(params, dtype, num_hidden_layers)
    nntile_model.clear_gradients()

    rng = np.random.default_rng(42)
    np_grad = np.array(
        rng.standard_normal(nntile_model.activations[-1].value.shape),
        dtype=dtype2np[dtype],
        order="F",
    )

    def bench_fn():
        # reset gradients each iteration to avoid accumulation/state issues
        nntile_model.clear_gradients()
        nntile_model.forward_async()
        nntile_model.activations[-1].grad.from_array(np_grad)
        nntile_model.backward_async()
        nntile.starpu.wait_for_all()

    nntile.starpu.wait_for_all()
    benchmark_model(bench_fn)
    nntile_model.unregister()
