# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/model/test_gpt2_attention.py
# Test for nntile.model.GPT2Attention
# Each test is generated in float precision by PyTorch, then it is downcasted
# into NNTile type. So, implementation of double precision is NOT checked.
#
# @version 1.1.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Config

import nntile
from nntile.model.gpt2_attention import GPT2Attention as GPT2Attention_nntile
from nntile.model.gpt2_config import GPT2ConfigNNTile
from nntile.tensor import TensorMoments, TensorTraits
from nntile.utils.constructors import to_numpy

# NNTile dtype via corresponding Tensor type
dtype2nntile = {
    "fp32": nntile.tensor.Tensor_fp32,
    "fp32_fast_tf32": nntile.tensor.Tensor_fp32_fast_tf32,
    "fp32_fast_fp16": nntile.tensor.Tensor_fp32_fast_fp16,
    "fp32_fast_bf16": nntile.tensor.Tensor_fp32_fast_bf16,
    "bf16": nntile.tensor.Tensor_bf16,
    "fp16": nntile.tensor.Tensor_fp16,
}

dtype2tol = {
    "fp32": {"rtol": 1e-6},
    "fp32_fast_tf32": {"rtol": 7e-4},
    "fp32_fast_fp16": {"rtol": 7e-4},
    "fp32_fast_bf16": {"rtol": 5e-3},
    "fp16": {"rtol": 5e-3},
    "bf16": {"rtol": 1.6e-2},
}

nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
flash_dtypes = {"fp16", "bf16"}


@dataclass
class GPT2AttentionTestParams:
    head_size: int
    n_head: int
    n_head_tile: int
    seq_len: int
    seq_len_tile: int
    n_batch: int
    n_batch_tile: int


single_tile = GPT2AttentionTestParams(
    head_size=64,
    n_head=4,
    n_head_tile=4,
    seq_len=64,
    seq_len_tile=64,
    n_batch=3,
    n_batch_tile=3,
)

multiple_tiles = GPT2AttentionTestParams(
    head_size=128,
    n_head=16,
    n_head_tile=8,
    seq_len=64,
    seq_len_tile=16,
    n_batch=4,
    n_batch_tile=1,
)


def generate_inputs(
    params: GPT2AttentionTestParams, dtype: str, flash_attention: bool = False
):
    rng = np.random.default_rng(42)
    hidden_size = params.head_size * params.n_head
    hidden_size_tile = params.head_size * params.n_head_tile
    torch_layer_config = GPT2Config(
        n_embd=hidden_size,
        n_head=params.n_head,
        use_cache=False,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
    )
    torch_layer = GPT2Attention(
        torch_layer_config, is_cross_attention=False, layer_idx=0
    )

    nntile_config = GPT2ConfigNNTile(
        vocab_size=torch_layer_config.vocab_size,
        vocab_embed_dim_tile=hidden_size,
        hidden_size=hidden_size,
        hidden_size_tile=hidden_size_tile,
        intermediate_size=torch_layer_config.n_inner,
        intermediate_size_tile=torch_layer_config.n_inner,
        n_head=params.n_head,
        n_head_tile=params.n_head_tile,
        dtype=dtype,
        flash_attention=flash_attention,
    )

    x_shape = [hidden_size, params.seq_len, params.n_batch]
    x_basetile = [
        hidden_size_tile,
        params.seq_len_tile,
        params.n_batch_tile,
    ]
    x_traits = TensorTraits(x_shape, x_basetile)
    x_distr = [0] * x_traits.grid.nelems
    x_type = dtype2nntile[dtype]
    x_value = x_type(x_traits, x_distr)
    x_grad = x_type(x_traits, x_distr)
    X = TensorMoments(x_value, x_grad, grad_required=True)

    x_random = rng.standard_normal(x_shape)
    x_nntile = np.array(x_random, dtype=np.float32, order="F")
    x_value.from_array(x_nntile)
    x_torch = torch.Tensor(x_nntile.T)
    x_torch.requires_grad_()

    nntile_layer = GPT2Attention_nntile.from_torch(
        torch_layer, X, nntile_config
    )
    nntile_layer.clear_gradients()

    y_grad_random = rng.standard_normal(x_shape)
    y_grad_nntile = np.array(y_grad_random, dtype=np.float32, order="F")
    nntile_layer.output.grad.from_array(y_grad_nntile)
    y_grad_torch = torch.Tensor(y_grad_nntile.T)

    return torch_layer, nntile_layer, x_torch, y_grad_torch


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(single_tile, id="single_tile"),
        pytest.param(multiple_tiles, id="multiple_tiles"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        "fp32",
        pytest.param("fp32_fast_tf32", marks=nocuda),
        pytest.param("fp32_fast_fp16", marks=nocuda),
        pytest.param("fp32_fast_bf16", marks=nocuda),
        pytest.param("bf16", marks=nocuda),
        pytest.param("fp16", marks=nocuda),
    ],
)
@pytest.mark.parametrize(
    "flash_attention",
    [
        False,
        pytest.param(True, marks=nocuda),
    ],
)
class TestGPT2Attention:
    def test_coercion(
        self, context, torch_rng, params: GPT2AttentionTestParams, dtype: str,
        flash_attention: bool
    ):
        if flash_attention and dtype not in flash_dtypes:
            pytest.skip("Flash attention requires fp16 or bf16 tensors")
        torch_layer, nntile_layer, *_ = generate_inputs(
            params, dtype, flash_attention=flash_attention
        )
        torch_layer_other = nntile_layer.to_torch()
        nntile_layer.unregister()

        rtol = dtype2tol[dtype]["rtol"]
        for (n1, p1), (n2, p2) in zip(
            torch_layer.named_parameters(),
            torch_layer_other.named_parameters()
        ):
            assert n1 == n2
            assert torch.norm(p1 - p2) <= rtol * torch.norm(p1)

    def test_forward(
        self, context, torch_rng, params: GPT2AttentionTestParams, dtype: str,
        flash_attention: bool
    ):
        if flash_attention and dtype not in flash_dtypes:
            pytest.skip("Flash attention requires fp16 or bf16 tensors")
        torch_layer, nntile_layer, x, _ = generate_inputs(
            params, dtype, flash_attention=flash_attention
        )
        y, _ = torch_layer(x)
        nntile_layer.forward_async()
        y_nntile = torch.Tensor(
            to_numpy(nntile_layer.output.value).T
        )
        nntile_layer.unregister()
        rtol = dtype2tol[dtype]["rtol"]
        assert torch.norm(y - y_nntile) <= rtol * torch.norm(y)

    def test_forward_backward(
        self, context, torch_rng, params: GPT2AttentionTestParams, dtype: str,
        flash_attention: bool
    ):
        if flash_attention and dtype not in flash_dtypes:
            pytest.skip("Flash attention requires fp16 or bf16 tensors")
        torch_layer, nntile_layer, x, y_grad = generate_inputs(
            params, dtype, flash_attention=flash_attention
        )
        y, _ = torch_layer(x)
        nntile_layer.forward_async()

        res = (y * y_grad).sum()
        res.backward()
        nntile_layer.backward_async()

        torch_layer_other = nntile_layer.to_torch_with_grads()
        grad_nntile = torch.Tensor(
            to_numpy(nntile_layer.activations[0].grad).T
        )
        nntile_layer.unregister()

        rtol = dtype2tol[dtype]["rtol"]
        assert torch.norm(x.grad - grad_nntile) <= rtol * torch.norm(x.grad)

        for (n1, p1), (n2, p2) in zip(
            torch_layer.named_parameters(),
            torch_layer_other.named_parameters()
        ):
            assert n1 == n2
            assert p1.requires_grad == p2.requires_grad
            if p1.requires_grad:
                g1, g2 = p1.grad, p2.grad
                assert torch.norm(g1 - g2) <= rtol * torch.norm(g1)
