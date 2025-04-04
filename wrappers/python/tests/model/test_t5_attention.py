# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/model/test_t5_attention.py
# Test for nntile.model.t5_attention - T5Attention
#
# @version 1.1.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from transformers.models.t5.modeling_t5 import (
    T5Attention as T5AttentionTorch,
    T5Config as T5ConfigTorch,
)

import nntile
from nntile.model.t5_config import T5ConfigNNTile
from nntile.layer.t5_attention import T5Attention, relative_position_bucket_numpy
from nntile.tensor import TensorMoments, TensorTraits
from transformers.models.t5.modeling_t5 import T5Attention as T5AttentionTorch
from nntile.utils.constructors import to_numpy

# NNTile dtype via corresponding Tensor type
dtype2nntile = {
    "fp32": nntile.tensor.Tensor_fp32,
    "fp32_fast_tf32": nntile.tensor.Tensor_fp32_fast_tf32,
    "bf16": nntile.tensor.Tensor_bf16,
}

dtype2tol = {
    "fp32": {"rtol": 3e-4},
    "fp32_fast_tf32": {"rtol": 7e-4},
    "bf16": {"rtol": 1.2e-2},
}

nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")


@dataclass
class T5AttentionTestParams:
    d_model: int
    d_model_tile: int
    d_ff: int
    d_ff_tile: int
    n_head: int
    n_head_tile: int
    n_batch: int
    n_batch_tile: int
    has_relative_bias: bool = True
    redux: bool = True
    seq_len: int = 100
    seq_len_tile: int = 100


multiple_tiles = T5AttentionTestParams(
    d_model=512,
    d_model_tile=128,
    d_ff=384,
    d_ff_tile=96,
    n_head=6,
    n_head_tile=2,
    seq_len=256,
    seq_len_tile=64,
    n_batch=4,
    n_batch_tile=1,
)

single_tile = T5AttentionTestParams(
    d_model=512,
    d_model_tile=512,
    d_ff=384,
    d_ff_tile=384,
    n_head=6,
    n_head_tile=6,
    seq_len=64,
    seq_len_tile=64,
    n_batch=3,
    n_batch_tile=3,
    has_relative_bias=True,
)


def generate_inputs(params: T5AttentionTestParams, dtype: str):
    # Configure PyTorch T5 layer
    torch_config = T5ConfigTorch(
        d_model=params.d_model,
        d_ff=params.d_ff,
        num_heads=params.n_head,
        dropout_rate=0.0,
    )
    torch_layer = T5AttentionTorch(torch_config, has_relative_attention_bias=params.has_relative_bias)

    # Configure NNTile T5 layer
    nntile_config = T5ConfigNNTile(
        d_model=params.d_model,
        d_model_tile=params.d_model_tile,
        d_ff=params.d_ff,
        d_ff_tile=params.d_ff_tile,
        n_head=params.n_head,
        n_head_tile=params.n_head_tile,
        redux=params.redux,
    )

    # Make sure all dropout layers are disabled
    for name, module in torch_layer.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

    torch_layer.eval()  # Set to evaluation mode

    # Set input tensor dimensions
    x_shape = [params.d_model, params.seq_len, params.n_batch]
    x_basetile = [params.d_model_tile, params.seq_len_tile, params.n_batch_tile]
    x_traits = TensorTraits(x_shape, x_basetile)
    x_distr = [0] * x_traits.grid.nelems
    x_type = dtype2nntile[dtype]
    x_value = x_type(x_traits, x_distr, 0)
    x_grad = x_type(x_traits, x_distr, 0)
    X = TensorMoments(x_value, x_grad, grad_required=True)

    # Generate random input data
    gen = np.random.default_rng(42)
    x_random = gen.standard_normal(x_shape, dtype=np.float32)
    x_nntile = np.array(x_random, dtype=np.float32, order="F")
    x_value.from_array(x_nntile)
    x_torch = torch.Tensor(x_nntile.T)
    x_torch.requires_grad_()

    # Initialize NNTile layer from PyTorch layer
    nntile_layer, _ = T5Attention.from_torch(torch_layer, X, None, nntile_config, 0)
    # nntile_layer.clear_gradients()

    # Generate random gradient for backward pass
    y_grad_random = gen.standard_normal(x_shape, dtype=np.float32)
    y_grad_nntile = np.array(y_grad_random, dtype=np.float32, order="F")
    nntile_layer.activations_output[0].grad.from_array(y_grad_nntile)
    y_grad_torch = torch.Tensor(y_grad_nntile.T)

    return torch_layer, nntile_layer, x_torch, y_grad_torch


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(single_tile, id="single_tile"),
        # pytest.param(multiple_tiles, id="multiple_tiles"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        "fp32",
        # pytest.param("fp32_fast_tf32", marks=nocuda),
        # pytest.param("bf16", marks=nocuda),
    ],
)
class TestT5Attention:
    def test_forward(
        self, starpu_simple, torch_rng, params: T5AttentionTestParams, dtype: str
    ):
        """Test that forward pass gives same results in PyTorch and NNTile"""
        torch_layer, nntile_layer, x, _ = generate_inputs(params, dtype)
        y, _, _ = torch_layer(x)
        nntile_layer.forward_async()
        y_nntile = torch.Tensor(to_numpy(nntile_layer.activations_output[0].value).T)
        # nntile_layer.unregister()
        rtol = dtype2tol[dtype]["rtol"]
        print("y: ", y)
        print("y_nntile: ", y_nntile)
        assert torch.norm(y - y_nntile) <= rtol * torch.norm(y)

    def test_backward(
        self, starpu_simple, torch_rng, params: T5AttentionTestParams, dtype: str
    ):
        """Test that backward pass gives same results in PyTorch and NNTile"""
        torch_layer, nntile_layer, x, y_grad = generate_inputs(params, dtype)

        y, _, _ = torch_layer(x)
        nntile_layer.forward_async()
        res = (y * y_grad).sum()
        res.backward()
        nntile_layer.backward_async()

        grad_nntile = torch.Tensor(to_numpy(nntile_layer.activations_input[0].grad).T)
        rtol = dtype2tol[dtype]["rtol"]
        
        assert torch.norm(x.grad - grad_nntile) <= rtol * torch.norm(x.grad)
        if params.has_relative_bias:
            nnt_bias_grad = torch.Tensor(to_numpy(nntile_layer.relative_bias_embedding.grad).T)
            assert torch.norm(nnt_bias_grad - torch_layer.relative_attention_bias.weight.grad) <= rtol * torch.norm(torch_layer.relative_attention_bias.weight.grad)
            
        nntile_layer.unregister()

    def test_relative_position_bucket(
        self, torch_rng, params: T5AttentionTestParams, dtype: str
    ):
        """Test relative position bucket calculation"""
        query_length, key_length = 3, 5
        context_position = np.arange(query_length, dtype=np.int32)[:, None]
        memory_position = np.arange(key_length, dtype=np.int32)[None, :]
        relative_position = memory_position - context_position

        # Test bidirectional case
        buckets_bidirectional = relative_position_bucket_numpy(
            relative_position,
            bidirectional=True,
            num_buckets=32,
            max_distance=128,
        )

        # Expected buckets for bidirectional case
        expected_buckets_bidirectional = np.array([
            [0, 17, 18, 19, 20],
            [1, 0, 17, 18, 19],
            [2, 1, 0, 17, 18]
        ], dtype=np.int32)

        np.testing.assert_array_equal(buckets_bidirectional, expected_buckets_bidirectional)

        # Test unidirectional case
        buckets_unidirectional = relative_position_bucket_numpy(
            relative_position,
            bidirectional=False,
            num_buckets=32,
            max_distance=128,
        )

        # Expected buckets for unidirectional case
        expected_buckets_unidirectional = np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [2, 1, 0, 0, 0]
        ], dtype=np.int32)
    
        np.testing.assert_array_equal(
            buckets_unidirectional, expected_buckets_unidirectional
        ) 
