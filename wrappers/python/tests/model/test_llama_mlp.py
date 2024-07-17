# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_llama_mlp.py
# Test for nntile.layer.LlamaMLP
# Each test is generated in float precision by PyTorch, then it is downcasted
# into NNTile type. So, implementation of double precision is NOT checked.
#
# @version 1.0.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaMLP

import nntile
from nntile.model.llama_config import LlamaConfigNNTile
from nntile.model.llama_mlp import LlamaMLP as LlamaMLP_nntile
# from nntile.model.llama import LlamaConfigNNTile
from nntile.tensor import TensorMoments, TensorTraits
from nntile.utils.constructors import to_numpy

# NNTile dtype via corresponding Tensor type
dtype2nntile = {
        'fp32': nntile.tensor.Tensor_fp32,
        'fp32_fast_tf32': nntile.tensor.Tensor_fp32_fast_tf32,
        'bf16': nntile.tensor.Tensor_bf16,
}

dtype2tol = {
        'fp32': {'rtol': 1e-6},
        'fp32_fast_tf32': {'rtol': 1e-4},
        'bf16': {'rtol': 1.6e-2},
}


def assert_close_by_frobnorm(a: np.ndarray, b: np.ndarray, rtol: float):
    np.testing.assert_array_less(
            np.linalg.norm(a - b),
            rtol * np.linalg.norm(a)
    )


@dataclass
class LlamaMLPTestParams:
    hidden_size: int
    hidden_size_tile: int
    intermediate_size: int
    intermediate_size_tile: int
    n_batch: int
    n_batch_tile: int
    dtype: str
    redux: bool = True
    activation_function: str = "silu"


TEST_PARAMS = [
    pytest.param(
        LlamaMLPTestParams(
            hidden_size=128,
            hidden_size_tile=32,
            intermediate_size=64,
            intermediate_size_tile=16,
            n_batch=4,
            n_batch_tile=1,
            dtype='bf16',
        ),
        marks=[
            pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is required"
            )
        ]
    ),
    pytest.param(
        LlamaMLPTestParams(
            hidden_size=128,
            hidden_size_tile=32,
            intermediate_size=64,
            intermediate_size_tile=16,
            n_batch=4,
            n_batch_tile=1,
            dtype='fp32_fast_tf32',
        ),
        marks=[
            pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is required"
            )
        ]
    ),
    LlamaMLPTestParams(
        hidden_size=128,
        hidden_size_tile=32,
        intermediate_size=64,
        intermediate_size_tile=16,
        n_batch=4,
        n_batch_tile=1,
        dtype='fp32',
    ),
    LlamaMLPTestParams(
        hidden_size=128,
        hidden_size_tile=32,
        intermediate_size=64,
        intermediate_size_tile=32,
        n_batch=4,
        n_batch_tile=2,
        dtype='fp32',
    ),
    LlamaMLPTestParams(
        hidden_size=128,
        hidden_size_tile=128,
        intermediate_size=64,
        intermediate_size_tile=64,
        n_batch=3,
        n_batch_tile=3,
        dtype='fp32',
    )
]


def generate_inputs(params: LlamaMLPTestParams):
    torch_layer_config = LlamaConfig(
        hidden_size=params.hidden_size,
        intermediate_size=params.intermediate_size,
        pretraining_tp=1
    )
    torch_layer = LlamaMLP(
        torch_layer_config
    )
    nntile_config = LlamaConfigNNTile(
        hidden_size=params.hidden_size,
        hidden_size_tile=params.hidden_size_tile,
        intermediate_size=params.intermediate_size,
        intermediate_size_tile=params.intermediate_size_tile,
    )
    x_shape = [params.hidden_size, params.n_batch]
    x_basetile = [params.hidden_size_tile, params.n_batch_tile]
    x_traits = TensorTraits(x_shape, x_basetile)
    x_distr = [0] * x_traits.grid.nelems
    x_type = dtype2nntile[params.dtype]
    x_value = x_type(x_traits, x_distr, 0)
    x_grad = x_type(x_traits, x_distr, 0)
    X = TensorMoments(x_value, x_grad, grad_required=True)
    gen = np.random.default_rng()
    x_random = gen.standard_normal(x_shape)
    x_nntile = np.array(x_random, dtype=np.float32, order="F")
    x_value.from_array(x_nntile)
    x_torch = torch.Tensor(x_nntile.T)
    nntile_layer, _ = LlamaMLP_nntile.from_torch(torch_layer, X,
                                                       nntile_config, 0)
    nntile_layer.clear_gradients()
    y_grad_random = gen.standard_normal(x_shape)
    y_grad_nntile = np.array(y_grad_random, dtype=np.float32, order="F")
    nntile_layer.activations[-1].grad.from_array(y_grad_nntile)
    y_grad_torch = torch.Tensor(y_grad_nntile.T)
    return torch_layer, nntile_layer, x_torch, y_grad_torch


@pytest.mark.parametrize("params", TEST_PARAMS)
class TestLlamaMLP:
    def test_from_torch_and_to_torch(
        self, starpu_simple, torch_rng, params: LlamaMLPTestParams
    ):
        torch_layer, nntile_layer, _, _ = generate_inputs(params)
        torch_layer_other = nntile_layer.to_torch()
        nntile_layer.unregister()
        assert_close_by_frobnorm(
            torch_layer.gate_proj.weight.detach().numpy(),
            torch_layer_other.gate_proj.weight.detach().numpy(),
            **dtype2tol[params.dtype]
        )
        assert_close_by_frobnorm(
            torch_layer.up_proj.weight.detach().numpy(),
            torch_layer_other.up_proj.weight.detach().numpy(),
            **dtype2tol[params.dtype]
        )
        assert_close_by_frobnorm(
            torch_layer.down_proj.weight.detach().numpy(),
            torch_layer_other.down_proj.weight.detach().numpy(),
            **dtype2tol[params.dtype]
        )

    def test_forward(
        self, starpu_simple, torch_rng, params: LlamaMLPTestParams
    ):
        torch_layer, nntile_layer, x, _ = generate_inputs(params)
        y = torch_layer(x)
        nntile_layer.forward_async()
        y_nntile = torch.Tensor(to_numpy(nntile_layer.activations[-1].value).T)
        nntile_layer.unregister()
        assert_close_by_frobnorm(
                y.detach().numpy(),
                y_nntile.detach().numpy(),
                **dtype2tol[params.dtype]
        )

    def test_forward_backward(
        self, starpu_simple, torch_rng, params: LlamaMLPTestParams
    ):
        torch_layer, nntile_layer, x, y_grad = generate_inputs(params)
        torch_layer_other = nntile_layer.to_torch()
        y = torch_layer(x)
        nntile_layer.forward_async()
        y_nntile = torch.Tensor(to_numpy(nntile_layer.activations[-1].value).T)
        # print(y.shape, y_nntile.shape)
        res = (y * y_grad).sum()
        res.backward()
        nntile_layer.backward_async()
        torch_layer_other = nntile_layer.to_torch_with_grads()
        nntile_layer.unregister()
        assert_close_by_frobnorm(
                y.detach().numpy(),
                y_nntile.detach().numpy(),
                **dtype2tol[params.dtype]
        )
        assert_close_by_frobnorm(
            torch_layer.up_proj.weight.grad.detach().numpy(),
            torch_layer_other.up_proj.weight.grad.detach().numpy(),
            **dtype2tol[params.dtype]
        )
        assert_close_by_frobnorm(
            torch_layer.gate_proj.weight.grad.detach().numpy(),
            torch_layer_other.gate_proj.weight.grad.detach().numpy(),
            **dtype2tol[params.dtype]
        )
        assert_close_by_frobnorm(
            torch_layer.down_proj.weight.grad.detach().numpy(),
            torch_layer_other.down_proj.weight.grad.detach().numpy(),
            **dtype2tol[params.dtype]
        )
