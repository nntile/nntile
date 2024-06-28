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
# @version 1.0.0

# All necesary imports
import nntile
import numpy as np
# Set up StarPU configuration and init it
config = nntile.starpu.Config(1, 0, 0)
# Init all NNTile-StarPU codelets
nntile.starpu.init()
# Define list of tested types
dtypes = [np.float32, np.float64]
# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
        np.float64: nntile.tensor.Tensor_fp64}
# Get LayerNorm from PyTorch
import torch
from torch.nn import LayerNorm

# Helper function returns bool value true if test passes
def helper(dtype: np.dtype):
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
    rand_A = np.random.randn(*A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A.value.from_array(np_A)
    torch_A = torch.tensor(np_A, requires_grad=True)
    rand_B_grad = np.random.randn(*A_shape)
    np_B_grad = np.array(rand_B_grad, dtype=dtype, order='F')
    rand_gamma = np.random.randn((A_shape[-1]))
    np_gamma = np.array(rand_gamma, dtype=dtype, order='F')
    rand_beta = np.random.randn((A_shape[-1]))
    np_beta = np.array(rand_beta, dtype=dtype, order='F')
    # Init NNTile LayerNorm
    nntile_layer, next_tag = nntile.layer.LayerNorm.generate_simple(A, \
            ndim-1, eps, next_tag)
    nntile_layer.gamma.value.from_array(np_gamma)
    nntile_layer.beta.value.from_array(np_beta)
    # Init PyTorch LayerNorm
    torch_layer = LayerNorm(A_shape[-1], eps=eps, dtype=torch_A.dtype)
    torch_layer.weight.data = torch.tensor(np_gamma)
    torch_layer.bias.data = torch.tensor(np_beta)
    # NNTile forward of LayerNorm
    nntile_layer.forward_async()
    np_B_nntile = np.zeros_like(np_A)
    nntile_layer.y.value.to_array(np_B_nntile)
    # PyTorch forward
    torch_B = torch_layer(torch_A)
    np_B_torch = torch_B.data.numpy()
    # Check forward
    if np.linalg.norm(np_B_torch-np_B_nntile) \
            / np.linalg.norm(np_B_torch) >= 1e-5:
        assert False
    # NNTile backward of LayerNorm
    nntile_layer.y.grad.from_array(np_B_grad)
    nntile.tensor.clear_async(nntile_layer.x.grad)
    nntile.tensor.clear_async(nntile_layer.gamma.grad)
    nntile.tensor.clear_async(nntile_layer.beta.grad)
    nntile_layer.backward_async()
    np_A_grad_nntile = np.zeros_like(np_A)
    nntile_layer.x.grad.to_array(np_A_grad_nntile)
    np_gamma_grad_nntile = np.zeros_like(np_gamma)
    nntile_layer.gamma.grad.to_array(np_gamma_grad_nntile)
    np_beta_grad_nntile = np.zeros_like(np_beta)
    nntile_layer.beta.grad.to_array(np_beta_grad_nntile)
    # PyTorch backward
    torch_B_grad = torch.tensor(np_B_grad, requires_grad=True)
    res = (torch_B*torch_B_grad).sum()
    res.backward()
    np_A_grad_torch = torch_A.grad.numpy()
    np_gamma_grad_torch = torch_layer.weight.grad.numpy()
    np_beta_grad_torch = torch_layer.bias.grad.numpy()
    # Check backward
    if np.linalg.norm(np_A_grad_torch-np_A_grad_nntile) \
            / np.linalg.norm(np_A_grad_torch) >= 1e-5:
        assert False
    if np.linalg.norm(np_gamma_grad_torch-np_gamma_grad_nntile) \
            / np.linalg.norm(np_gamma_grad_torch) >= 1e-5:
        assert False
    if np.linalg.norm(np_beta_grad_torch-np_beta_grad_nntile) \
            / np.linalg.norm(np_beta_grad_torch) >= 1e-5:
        assert False
    # Unregister tensors
    A.unregister()
    return True

# Test runner for different precisions
def test():
    for dtype in dtypes:
        assert helper(dtype)

# Repeat tests
def test_repeat():
    for dtype in dtypes:
        assert helper(dtype)

if __name__ == "__main__":
    test()
    test_repeat()
