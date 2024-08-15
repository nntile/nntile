# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_layer_norm.py
# Test for nntile.layer.LayerNorm
#
# @version 1.1.0

import numpy as np
import pytest
import torch
import torch.nn as nn

import nntile
import nntile.utils.constructors as nntc

# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
          np.float64: nntile.tensor.Tensor_fp64}


config = nntile.starpu.Config(1, 0, 0)
nntile.starpu.init()


class RMSNorm(torch.nn.Module):
    """See `original`_ implementation.

    .. _original: https://github.com/meta-llama/llama/blob/main/llama/model.py`
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator
                                   for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator
                         for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_rms_norm(dtype: np.dtype):
    # Describe single-tile tensor, located at node 0
    A_shape = [20, 30]
    ndim = len(A_shape)
    eps = 1e-5
    A_traits = nntile.tensor.TensorTraits(A_shape, A_shape)
    mpi_distr = [0]
    next_tag = 0
    # Tensor objects
    A_value = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A_value.next_tag
    A_grad = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A_grad.next_tag
    A = nntile.tensor.TensorMoments(A_value, A_grad, True)
    # Set initial values of tensors
    gen = np.random.default_rng()
    rand_A = gen.standard_normal(size=A_shape, dtype=np.float32)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A.value.from_array(np_A)
    torch_A = torch.tensor(np_A, requires_grad=True)
    rand_B_grad = gen.standard_normal(size=A_shape, dtype=np.float32)
    np_B_grad = np.array(rand_B_grad, dtype=dtype, order='F')
    rand_gamma = gen.standard_normal(size=A_shape[-1], dtype=np.float32)
    np_gamma = np.array(rand_gamma, dtype=dtype, order='F')
    # Init NNTile LayerNorm
    nntile_layer, next_tag = nntile.layer.RMSNorm.generate_simple(A,
            ndim - 1, eps, next_tag)
    nntile_layer.gamma.value.from_array(np_gamma)
    # Init PyTorch LayerNorm
    torch_layer = RMSNorm(A_shape[-1], eps=eps)
    torch_layer.weight.data = torch.tensor(np_gamma)
    # NNTile forward of LayerNorm
    nntile_layer.forward_async()
    np_B_nntile = np.zeros_like(np_A)
    nntile_layer.y.value.to_array(np_B_nntile)
    # PyTorch forward
    torch_B = torch_layer(torch_A)
    np_B_torch = torch_B.data.numpy()
    # Check forward
    assert (np.linalg.norm(np_B_torch - np_B_nntile) /
            np.linalg.norm(np_B_torch) <= 1e-5)
    # NNTile backward of LayerNorm
    nntile_layer.y.grad.from_array(np_B_grad)
    nntile.tensor.clear_async(nntile_layer.x.grad)
    nntile.tensor.clear_async(nntile_layer.gamma.grad)
    nntile_layer.backward_async()
    np_A_grad_nntile = np.zeros_like(np_A)
    nntile_layer.x.grad.to_array(np_A_grad_nntile)
    np_gamma_grad_nntile = np.zeros_like(np_gamma)
    nntile_layer.gamma.grad.to_array(np_gamma_grad_nntile)
    # PyTorch backward
    torch_B_grad = torch.tensor(np_B_grad, requires_grad=True)
    res = (torch_B * torch_B_grad).sum()
    res.backward()
    np_A_grad_torch = torch_A.grad.numpy()
    np_gamma_grad_torch = torch_layer.weight.grad.numpy()
    # Check backward
    assert (np.linalg.norm(np_A_grad_torch - np_A_grad_nntile)
            / np.linalg.norm(np_A_grad_torch) <= 1e-5)
    assert (np.linalg.norm(np_gamma_grad_torch - np_gamma_grad_nntile) /
            np.linalg.norm(np_gamma_grad_torch) <= 1e-5)

    # Unregister tensors
    A.unregister()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rms_norm_dynamic(numpy_rng, dtype: np.dtype):
    # Describe single-tile tensor, located at node 0
    A_shape = [20, 30]
    ndim = len(A_shape)
    eps = 1e-5

    # Generate input
    np_A = np.asfortranarray(
        numpy_rng.standard_normal(size=A_shape, dtype=np.float32)
    )
    np_gamma = np.asfortranarray(
        numpy_rng.standard_normal(size=A_shape[-1], dtype=np.float32)
    )

    A = nntile.tensor.TensorMoments(nntc.from_array(np_A), None, False)
    torch_A = torch.tensor(np_A)

    # Init NNTile LayerNorm
    nntile_layer, _ = nntile.layer.RMSNorm.generate_simple(A, ndim - 1, eps, 0)
    nntile_layer.gamma.value.from_array(np_gamma)
    # Init PyTorch LayerNorm
    torch_layer = RMSNorm(A_shape[-1], eps=eps)
    torch_layer.weight.data = torch.tensor(np_gamma)

    # NNTile forward of LayerNorm
    B_nntile = nntile_layer.forward_dynamic(A)
    np_B_nntile = nntc.to_numpy(B_nntile.value)
    # PyTorch forward
    torch_B = torch_layer(torch_A)
    np_B_torch = torch_B.data.numpy()
    # Check forward
    assert (
        np.linalg.norm(np_B_torch - np_B_nntile) / np.linalg.norm(np_B_torch)
        <= 1e-5
    )

    np_A_trunc = np_A[: np_A.shape[0] // 2, :]

    B_nntile = nntile_layer.forward_dynamic(
        nntile.tensor.TensorMoments(nntc.from_array(np_A_trunc), None, False)
    )
    np_B_nntile = nntc.to_numpy(B_nntile.value)
    # PyTorch forward
    torch_B = torch_layer(torch.tensor(np_A_trunc))
    np_B_torch = torch_B.data.numpy()
    # Check forward
    assert (
        np.linalg.norm(np_B_torch - np_B_nntile) / np.linalg.norm(np_B_torch)
        <= 1e-5
    )

    # Unregister tensors
    A.unregister()
