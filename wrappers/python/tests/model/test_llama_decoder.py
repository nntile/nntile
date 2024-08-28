# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_llama_decoder.py
# Test for nntile.model.Llamadecoder
# Each test is generated in float precision by PyTorch, then it is downcasted
# into NNTile type. So, implementation of double precision is NOT checked.
#
# @version 1.1.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaModel

import nntile
from nntile.model.llama_config import LlamaConfigNNTile
from nntile.model.llama_decoder import LlamaDecoder as LlamaDecoder_nntile
from nntile.tensor import TensorMoments, TensorTraits
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
class LlamaDecoderTestParams:
    hidden_size: int
    hidden_size_tile: int
    intermediate_size: int
    intermediate_size_tile: int
    n_batch: int
    n_batch_tile: int
    redux: bool = True
    activation_function: str = "silu"
    seq_len: int = 100
    seq_len_tile: int = 100


single_tile = LlamaDecoderTestParams(
    hidden_size=128,
    hidden_size_tile=128,
    intermediate_size=64,
    intermediate_size_tile=64,
    seq_len=32,
    seq_len_tile=32,
    n_batch=3,
    n_batch_tile=3)

multiple_tiles = LlamaDecoderTestParams(
    hidden_size=128,
    hidden_size_tile=32,
    intermediate_size=64,
    intermediate_size_tile=16,
    seq_len=128,
    seq_len_tile=32,
    n_batch=4,
    n_batch_tile=1)


def generate_inputs(params: LlamaDecoderTestParams,
                    dtype: str,
                    att_bias: bool,
                    flash_attention: bool):
    torch_layer_config = LlamaConfig(
        hidden_size=params.hidden_size,
        intermediate_size=params.intermediate_size,
        pretraining_tp=1,
        num_hidden_layers=1,
        attention_bias=att_bias
    )
    llama_torch = LlamaModel(torch_layer_config)
    torch_layer = llama_torch.layers[0]
    nntile_config = LlamaConfigNNTile(
        vocab_size=torch_layer_config.vocab_size,
        vocab_embed_dim_tile=params.hidden_size,
        max_position_embeddings=torch_layer_config.max_position_embeddings,
        n_attention_head=torch_layer_config.num_attention_heads,
        n_head_tile=torch_layer_config.num_attention_heads,
        num_key_value_heads=torch_layer_config.num_key_value_heads,
        hidden_size=params.hidden_size,
        hidden_size_tile=params.hidden_size_tile,
        intermediate_size=params.intermediate_size,
        intermediate_size_tile=params.intermediate_size_tile,
        dtype=dtype,
        attention_bias=att_bias,
        flash_attention=flash_attention
    )
    x_shape = [params.hidden_size, params.seq_len, params.n_batch]
    x_basetile = [params.hidden_size_tile,
                  params.seq_len_tile,
                  params.n_batch_tile]
    x_traits = TensorTraits(x_shape, x_basetile)
    x_distr = [0] * x_traits.grid.nelems
    x_type = dtype2nntile[dtype]
    x_value = x_type(x_traits, x_distr, 0)
    x_grad = zeros_like(x_value)
    X = TensorMoments(x_value, x_grad, grad_required=True)
    gen = np.random.default_rng(42)
    x_random = gen.standard_normal(x_shape, dtype=np.float32)
    x_nntile = np.array(x_random, dtype=np.float32, order="F")
    x_value.from_array(x_nntile)
    x_torch = torch.tensor(x_nntile.T, requires_grad=True)
    pos_ids = gen.integers(params.seq_len,
                           size=(params.n_batch, params.seq_len),
                           dtype=np.int64)
    mask = np.array(np.triu(np.ones((params.seq_len, params.seq_len))),
                    dtype=bool, order="F")
    nntile_layer, _ = LlamaDecoder_nntile.from_torch(torch_layer, X,
                                                     pos_ids, mask,
                                                     nntile_config, 0)
    nntile_layer.clear_gradients()
    y_grad_random = gen.standard_normal(x_shape, dtype=np.float32)
    y_grad_nntile = np.array(y_grad_random, dtype=np.float32, order="F")
    nntile_layer.activations[-1].grad.from_array(y_grad_nntile)
    y_grad_torch = torch.Tensor(y_grad_nntile.T)
    return torch_layer, nntile_layer, x_torch, y_grad_torch, pos_ids, mask


@pytest.mark.parametrize('params', [
    pytest.param(single_tile, id='single_tile'),
    pytest.param(multiple_tiles, id='multiple_tiles'),
])
@pytest.mark.parametrize('dtype', [
    'fp32',
    pytest.param('fp32_fast_tf32', marks=nocuda),
    pytest.param('bf16', marks=nocuda),
])
@pytest.mark.parametrize('att_bias', [
    False,
    # True # Temporarily disabled to investigate later
])
@pytest.mark.parametrize('flash_attention', [True, False])
class TestLlamaMLP:
    def test_coercion(self, starpu_simple, torch_rng,
                      params: LlamaDecoderTestParams, dtype: str,
                      att_bias: bool, flash_attention: bool):
        torch_layer, nntile_layer, *_ = generate_inputs(
            params, dtype, att_bias, flash_attention)
        torch_layer_other = nntile_layer.to_torch()
        nntile_layer.unregister()

        rtol = dtype2tol[dtype]['rtol']
        for (n1, p1), (n2, p2) in zip(torch_layer.named_parameters(),
                torch_layer_other.named_parameters()):
            assert n1 == n2
            assert torch.norm(p1 - p2) <= rtol * torch.norm(p1)

    def test_forward(self, starpu_simple, torch_rng,
                     params: LlamaDecoderTestParams,
                     dtype: str,
                     att_bias: bool,
                     flash_attention: bool):
        torch_layer, nntile_layer, x, _, pos_ids, mask = generate_inputs(
                params, dtype, att_bias, flash_attention)
        mask_torch = torch.Tensor(np.array(1 - mask, dtype=np.float32)).T \
            * torch.finfo(torch.float32).min
        mask_torch = mask_torch[None, None, :, :].expand(params.n_batch,
                                                         1, -1, -1)
        y = torch_layer(x, position_ids=torch.tensor(pos_ids),
                        attention_mask=mask_torch)[0]
        nntile_layer.forward_async()
        y_nntile = torch.Tensor(to_numpy(nntile_layer.activations[-1].value).T)
        nntile_layer.unregister()
        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(y - y_nntile) <= rtol * torch.norm(y)

    def test_backward(self, starpu_simple, torch_rng,
                      params: LlamaDecoderTestParams, dtype: str,
                      att_bias: bool, flash_attention: bool):
        torch_layer, nntile_layer, x, y_grad, pos_ids, mask = \
            generate_inputs(params, dtype, att_bias, flash_attention)
        torch_layer_other = nntile_layer.to_torch()
        mask_torch = torch.Tensor(np.array(1 - mask, dtype=np.float32)).T \
            * torch.finfo(torch.float32).min
        mask_torch = mask_torch[None, None, :, :].expand(params.n_batch,
                                                         1, -1, -1)
        y = torch_layer(x, position_ids=torch.tensor(pos_ids),
                        attention_mask=mask_torch)[0]
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
