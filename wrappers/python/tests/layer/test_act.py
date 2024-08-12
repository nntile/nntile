# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_act.py
# Test for nntile.layer.act
#
# @version 1.1.0

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.autograd.functional import jvp

import nntile
import nntile.tensor
import nntile.utils.constructors as nntc
from nntile.layer import Act

config = nntile.starpu.Config(1, 0, 0)
nntile.starpu.init()

# Define mapping between numpy and nntile types
Tensor = {
    np.float32: nntile.tensor.Tensor_fp32,
    np.float64: nntile.tensor.Tensor_fp64,
}


def setup(name: str, dtype: np.dtype):
    # Describe single-tile tensor, located at node 0
    A_shape = [4, 5, 6]
    A_traits = nntile.tensor.TensorTraits(A_shape, A_shape)
    mpi_distr = [0]
    next_tag = 0
    # Tensor objects
    A = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    A_grad = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A_grad.next_tag
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)
    # Set initial values of tensors
    rand_A = np.random.default_rng(42).standard_normal(A_shape)
    np_A = np.array(rand_A, dtype=dtype, order="F")
    np_B = np.zeros_like(np_A)

    # A is invalidated after each forward_async
    A.from_array(np_A)
    nntile.tensor.clear_async(A_grad)
    # Set up activation layer
    layer, _ = Act.generate_simple(A_moments, name, next_tag)

    return np_A, np_B, A_moments, layer


@pytest.mark.parametrize("name", ["relu", "gelu", "gelutanh", "silu"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
class TestAct:
    def test_forward(self, name: str, dtype: np.dtype):
        if dtype == np.float32:
            tol = 1e-5
        elif dtype == np.float64:
            tol = 1e-10
        np_A, np_B, A_moments, layer = setup(name, dtype)

        # Do forward pass and wait until it is finished
        layer.forward_async()
        nntile.starpu.wait_for_all()
        # Dump output
        layer.y.value.to_array(np_B)
        # Check output correctness
        if name == "relu":
            torch_output = F.relu(torch.from_numpy(np_A))
        elif name == "gelu":
            torch_output = F.gelu(torch.from_numpy(np_A))
        elif name == "gelutanh":
            torch_output = F.gelu(torch.from_numpy(np_A), approximate="tanh")
        elif name == "silu":
            torch_output = F.silu(torch.from_numpy(np_A))
        np_C = np.array(torch_output.numpy(), order="F", dtype=dtype)
        assert np.linalg.norm(np_C - np_B) / np.linalg.norm(np_C) <= tol

        A_moments.unregister()
        layer.unregister()

    def test_backward(self, name: str, dtype: np.dtype):
        if dtype == np.float32:
            tol = 1e-5
        elif dtype == np.float64:
            tol = 1e-10
        np_A, np_B, A_moments, layer = setup(name, dtype)

        # Do backward
        layer.y.grad.from_array(2 * np_A)
        layer.backward_async()
        nntile.starpu.wait_for_all()
        # Dump output
        layer.x.grad.to_array(np_B)
        # Check correctness
        if name == "relu":
            torch_grad = jvp(
                F.relu, torch.from_numpy(np_A), torch.from_numpy(2 * np_A)
            )[1]
        elif name == "gelu":
            torch_grad = jvp(
                F.gelu, torch.from_numpy(np_A), torch.from_numpy(2 * np_A)
            )[1]
        elif name == "gelutanh":
            torch_grad = jvp(
                lambda x: F.gelu(x, approximate="tanh"),
                torch.from_numpy(np_A),
                torch.from_numpy(2 * np_A),
            )[1]
        elif name == "silu":
            torch_grad = jvp(
                lambda x: F.silu(x),
                torch.from_numpy(np_A),
                torch.from_numpy(2 * np_A),
            )[1]
        np_C = np.array(torch_grad.numpy(), order="F", dtype=dtype)
        assert np.linalg.norm(np_C - np_B) / np.linalg.norm(np_C) <= tol

        A_moments.unregister()
        layer.unregister()

    def test_dynamic(self, name: str, dtype: np.dtype):
        if dtype == np.float32:
            tol = 1e-5
        elif dtype == np.float64:
            tol = 1e-10
        np_A, np_B, A_moments, layer = setup(name, dtype)

        # Do forward pass and wait until it is finished
        input_nnt = nntc.from_array(np_A)
        out_dyn = layer.forward_dynamic(
            nntile.tensor.TensorMoments(input_nnt, None, False)
        )
        nntile.starpu.wait_for_all()
        # Dump output
        np_B = nntc.to_numpy(out_dyn.value)
        # Check output correctness
        if name == "relu":
            torch_output = F.relu(torch.from_numpy(np_A))
        elif name == "gelu":
            torch_output = F.gelu(torch.from_numpy(np_A))
        elif name == "gelutanh":
            torch_output = F.gelu(torch.from_numpy(np_A), approximate="tanh")
        elif name == "silu":
            torch_output = F.silu(torch.from_numpy(np_A))
        np_C = np.array(torch_output.numpy(), order="F", dtype=dtype)
        assert np.linalg.norm(np_C - np_B) / np.linalg.norm(np_C) <= tol

        A_moments.unregister()
        layer.unregister()
