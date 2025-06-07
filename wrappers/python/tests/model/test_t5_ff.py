# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/model/test_t5_ff.py
# Test for nntile.model.t5_ff - T5LayerFF
#
# @version 1.1.0
# ruff: noqa: E501

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from transformers.models.t5.modeling_t5 import (
    T5Config as T5ConfigTorch, T5LayerFF as T5LayerFFTorch)

import nntile
from nntile.model.t5_config import T5ConfigNNTile
from nntile.model.t5_ff import T5LayerFF
from nntile.tensor import TensorMoments, TensorTraits
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
class T5FFTestParams:
    d_model: int
    d_model_tile: int
    d_ff: int
    d_ff_tile: int
    n_batch: int
    n_batch_tile: int
    redux: bool = False  # Disabled because it causes SegFaults
    seq_len: int = 100
    seq_len_tile: int = 100


single_tile = T5FFTestParams(
    d_model=128,
    d_model_tile=128,
    d_ff=128,
    d_ff_tile=128,
    seq_len=64,
    seq_len_tile=64,
    n_batch=2,
    n_batch_tile=2,
)

multiple_tiles = T5FFTestParams(
    d_model=128,
    d_model_tile=32,
    d_ff=128,
    d_ff_tile=64,
    seq_len=64,
    seq_len_tile=16,
    n_batch=4,
    n_batch_tile=1,
)


def generate_inputs(params: T5FFTestParams, dtype: str):
    # Configure PyTorch T5 layer
    torch_config = T5ConfigTorch(
        d_model=params.d_model,
        d_ff=params.d_ff,
        dropout_rate=0.0,
        dense_act_fn="gelu_new",
        is_gated_act=True,
    )
    torch_layer = T5LayerFFTorch(torch_config)

    # Configure NNTile T5 layer
    nntile_config = T5ConfigNNTile(
        d_model=params.d_model,
        d_model_tile=params.d_model_tile,
        d_ff=params.d_ff,
        d_ff_tile=params.d_ff_tile,
        dense_act_fn="gelu",
        dropout_rate=0.0,
        is_gated_act=True,
        redux=params.redux,
        # Not used in T5LayerFF
        d_kv=None,
        d_kv_tile=None,
        n_head=None,
        n_head_tile=None,
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
    x_value = x_type(x_traits, x_distr)
    x_grad = x_type(x_traits, x_distr)
    X = TensorMoments(x_value, x_grad, grad_required=True)

    # Generate random input data
    gen = np.random.default_rng(42)
    x_random = gen.standard_normal(x_shape, dtype=np.float32)
    x_nntile = np.array(x_random, dtype=np.float32, order="F")
    x_value.from_array(x_nntile)
    x_torch = torch.Tensor(x_nntile.T)
    x_torch.requires_grad_()

    # Initialize NNTile layer from PyTorch layer
    nntile_layer = T5LayerFF.from_torch(torch_layer, X, nntile_config)
    nntile_layer.clear_gradients()

    # Generate random gradient for backward pass
    y_grad_random = gen.standard_normal(x_shape, dtype=np.float32)
    y_grad_nntile = np.array(y_grad_random, dtype=np.float32, order="F")
    nntile_layer.activations[-1].grad.from_array(y_grad_nntile)
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
        pytest.param("bf16", marks=nocuda),
    ],
)
class TestT5LayerFF:

    def test_coercion(
        self, context, torch_rng, params: T5FFTestParams, dtype: str
    ):
        """Test that weights from PyTorch can be converted to NNTile and back"""
        torch_layer, nntile_layer, _, _ = generate_inputs(params, dtype)
        torch_layer_other = nntile_layer.to_torch()
        nntile_layer.unregister()

        rtol = dtype2tol[dtype]["rtol"]
        for (n1, p1), (n2, p2) in zip(
            torch_layer.named_parameters(), torch_layer_other.named_parameters()
        ):
            assert n1 == n2
            assert torch.norm(p1 - p2) <= rtol * torch.norm(p1)

    def test_forward(
        self, context, torch_rng, params: T5FFTestParams, dtype: str
    ):
        """Test that forward pass gives same results in PyTorch and NNTile"""
        torch_layer, nntile_layer, x, _ = generate_inputs(params, dtype)
        y = torch_layer(x)
        nntile_layer.forward_async()
        y_nntile = torch.Tensor(to_numpy(nntile_layer.activations[-1].value).T)
        nntile_layer.unregister()
        rtol = dtype2tol[dtype]["rtol"]
        assert torch.norm(y - y_nntile) <= rtol * torch.norm(y)

    def test_backward(
        self, context, torch_rng, params: T5FFTestParams, dtype: str
    ):
        """Test that backward pass gives same results in PyTorch and NNTile"""
        torch_layer, nntile_layer, x, y_grad = generate_inputs(params, dtype)
        torch_layer_other = nntile_layer.to_torch()
        y = torch_layer(x)
        nntile_layer.forward_async()
        res = (y * y_grad).sum()
        res.backward()
        nntile_layer.backward_async()
        torch_layer_other = nntile_layer.to_torch_with_grads()
        grad_nntile = torch.Tensor(to_numpy(nntile_layer.activations[0].grad).T)
        nntile_layer.unregister()
        rtol = dtype2tol[dtype]["rtol"]
        assert torch.norm(x.grad - grad_nntile) <= rtol * torch.norm(x.grad)

        for (n1, p1), (n2, p2) in zip(
            torch_layer.named_parameters(), torch_layer_other.named_parameters()
        ):
            assert n1 == n2
            assert p1.requires_grad == p2.requires_grad
            if p1.requires_grad:
                g1, g2 = p1.grad, p2.grad
                assert torch.norm(g1 - g2) <= rtol * torch.norm(g1)
