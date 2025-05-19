# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/model/test_t5_lmhead.py
# Test for nntile.model.t5_lmhead - T5ClassificationHead
#
# @version 1.1.0

from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn as nn
# Import the official HuggingFace implementation
from transformers.models.t5.modeling_t5 import T5Config as T5ConfigTorch

import nntile
import nntile.utils.constructors as nntc
from nntile.model.t5_config import T5ConfigNNTile
from nntile.model.t5_lmhead import T5ClassificationHead
from nntile.tensor import TensorMoments, TensorTraits

# NNTile dtype via corresponding Tensor type
dtype2nntile = {
    "fp32": nntile.tensor.Tensor_fp32,
    "fp32_fast_tf32": nntile.tensor.Tensor_fp32_fast_tf32,
    "bf16": nntile.tensor.Tensor_bf16,
}

dtype2tol = {
    "fp32": {"atol": 5e-4},
    "fp32_fast_tf32": {"atol": 5e-4},
    "bf16": {"atol": 1e-2},
}

nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")


@dataclass
class T5ClassificationHeadTestParams:
    d_model: int
    d_model_tile: int
    num_labels: int
    n_batch: int
    n_batch_tile: int
    redux: bool = True
    seq_len: int = 100
    seq_len_tile: int = 100


single_tile = T5ClassificationHeadTestParams(
    d_model=128,
    d_model_tile=128,
    num_labels=10,
    seq_len=64,
    seq_len_tile=64,
    n_batch=3,
    n_batch_tile=3,
)

multiple_tiles = T5ClassificationHeadTestParams(
    d_model=128,
    d_model_tile=32,
    num_labels=10,
    seq_len=128,
    seq_len_tile=32,
    n_batch=4,
    n_batch_tile=1,
)


class T5ClassificationHeadTest(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


def generate_inputs(params: T5ClassificationHeadTestParams, dtype: str):
    """Generate test inputs for both PyTorch and NNTile models"""
    # Create an official PyTorch T5ClassificationHead
    torch_config = T5ConfigTorch(
        d_model=params.d_model,
        num_labels=params.num_labels,
        dropout_rate=0.0,  # No dropout for testing
    )
    # torch_head = T5ClassificationHeadTorch(torch_config)
    torch_head = T5ClassificationHeadTest(torch_config)
    torch_head.eval()  # Set to evaluation mode

    # Configure NNTile model config
    nntile_config = T5ConfigNNTile(
        d_model=params.d_model,
        d_model_tile=params.d_model_tile,
        d_kv=8,  # Not used by classification head but required by config
        d_kv_tile=8,
        d_ff=256,  # Not used by classification head but required by config
        d_ff_tile=256,
        n_head=4,  # Not used by classification head but required by config
        n_head_tile=4,
        redux=params.redux,
    )

    # Generator for random values
    gen = np.random.default_rng(42)

    # Set input tensor dimensions
    x_shape = [params.d_model, params.seq_len, params.n_batch]
    x_basetile = [params.d_model_tile, params.seq_len_tile, params.n_batch_tile]
    x_traits = TensorTraits(x_shape, x_basetile)
    x_distr = [0] * x_traits.grid.nelems
    x_type = dtype2nntile[dtype]
    x_value = x_type(x_traits, x_distr, 0)
    x_grad = x_type(x_traits, x_distr, 0)
    x = TensorMoments(x_value, x_grad, grad_required=True)
    nntile.functions.fill_async(0.0, x.grad)

    # Generate random input data
    x_random = gen.standard_normal(x_shape, dtype=np.float32)
    x_nntile = np.array(x_random, dtype=np.float32, order="F")
    x_value.from_array(x_nntile)
    x_torch = torch.Tensor(x_nntile.T)
    x_torch.requires_grad_()

    # Create NNTile model
    nntile_head, _ = T5ClassificationHead.from_torch(
        torch_head, x, nntile_config, params.num_labels, next_tag=0
    )
    nntile_head.clear_gradients()

    # Generate random gradient for backward pass
    y_grad_random = gen.standard_normal(
        [params.num_labels, params.seq_len, params.n_batch], dtype=np.float32
    )
    y_grad_nntile = np.array(y_grad_random, dtype=np.float32, order="F")
    nntile_head.activations[-1].grad.from_array(y_grad_nntile)
    y_grad_torch = torch.Tensor(y_grad_nntile.T)

    return torch_head, nntile_head, x_torch, y_grad_torch


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
class TestT5ClassificationHead:
    def test_forward(
        self, starpu_simple, params: T5ClassificationHeadTestParams, dtype: str
    ):
        """Test that forward pass gives same results in PyTorch and NNTile"""
        torch_head, nntile_head, x_torch, _ = generate_inputs(params, dtype)

        # PyTorch forward pass
        y_torch = torch_head(x_torch)

        # NNTile forward pass
        nntile_head.forward_async()
        # nntile.starpu.wait_for_all()
        y_nntile = torch.Tensor(nntc.to_numpy(nntile_head.activations[-1].value).T)

        # Check if results match
        np.testing.assert_allclose(
            y_torch.detach().numpy(), y_nntile.detach().numpy(), **dtype2tol[dtype]
        )

    def test_backward(
        self, starpu_simple, params: T5ClassificationHeadTestParams, dtype: str
    ):
        """Test that backward pass gives same results in PyTorch and NNTile"""
        torch_head, nntile_head, x_torch, y_grad_torch = generate_inputs(params, dtype)

        # PyTorch forward and backward pass
        y_torch = torch_head(x_torch)
        y_torch.backward(y_grad_torch)

        # NNTile forward and backward pass
        nntile_head.forward_async()
        nntile_head.backward_async()

        # Compare gradients for input
        x_grad_nntile = torch.Tensor(nntc.to_numpy(nntile_head.activations[0].grad).T)
        np.testing.assert_allclose(
            x_torch.grad.detach().numpy(),
            x_grad_nntile.detach().numpy(),
            **dtype2tol[dtype],
        )
