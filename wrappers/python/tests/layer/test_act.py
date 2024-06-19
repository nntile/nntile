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
# @version 1.0.0

# All necesary imports
import nntile
import numpy as np
import torch.nn.functional as F
import torch
# Set up StarPU configuration and init it
config = nntile.starpu.Config(1, 0, 0)
# Init all NNTile-StarPU codelets
nntile.starpu.init()
# Define list of tested types
dtypes = [np.float32, np.float64]
# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
        np.float64: nntile.tensor.Tensor_fp64}
# Get multiprecision activation layer
Act = nntile.layer.Act

# Helper function returns bool value true if test passes
def helper(dtype: np.dtype):
    if dtype == np.float32:
        tol = 1e-5
    elif dtype == np.float64:
        tol = 1e-10
    # Describe single-tile tensor, located at node 0
    A_shape = [4, 5, 6]
    ndim = len(A_shape)
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
    rand_A = np.random.randn(*A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    np_B = np.zeros_like(np_A)
    # Check result for each activation function
    for funcname in Act.activations:
        # A is invalidated after each forward_async
        A.from_array(np_A)
        nntile.tensor.clear_async(A_grad)
        # Set up activation layer
        layer, next_tag = Act.generate_simple(A_moments, funcname, next_tag)
        # Do forward pass and wait until it is finished
        layer.forward_async()
        nntile.starpu.wait_for_all()
        # Dump output
        layer.y.value.to_array(np_B)
        # Check output correctness
        if funcname == "relu":
            torch_output = F.relu(torch.from_numpy(np_A))
        elif funcname == "gelu":
            torch_output = F.gelu(torch.from_numpy(np_A))
        elif funcname == "gelutanh":
            torch_output = F.gelu(torch.from_numpy(np_A), approximate="tanh")
        np_C = np.array(torch_output.numpy(), order="F", dtype=dtype)
        if np.linalg.norm(np_C - np_B) / np.linalg.norm(np_C) > tol:
            A_moments.unregister()
            layer.unregister()
            assert False
        # Do backward
        layer.y.grad.from_array(2*np_A)
        layer.backward_async()
        nntile.starpu.wait_for_all()
        # Dump output
        layer.x.grad.to_array(np_B)
        # Check correctness
        if funcname == "relu":
            torch_grad = torch.autograd.functional.jvp(F.relu, \
                    torch.from_numpy(np_A), torch.from_numpy(2 * np_A))[1]
        elif funcname == "gelu":
            torch_grad = torch.autograd.functional.jvp(F.gelu, \
                    torch.from_numpy(np_A), torch.from_numpy(2 * np_A))[1]
        elif funcname == "gelutanh":
            torch_grad = torch.autograd.functional.jvp( \
                    lambda x: F.gelu(x, approximate="tanh"), \
                    torch.from_numpy(np_A), torch.from_numpy(2 * np_A))[1]
        np_C = np.array(torch_grad.numpy(), order="F", dtype=dtype)
        if np.linalg.norm(np_C - np_B) / np.linalg.norm(np_C) > tol:
            A_moments.unregister()
            layer.unregister()
            assert False
    A_moments.unregister()
    layer.unregister()
    print("Finish checking {}".format(Act.activations.keys()))
    assert True

# Test runner for different precisions
def test():
    for dtype in dtypes:
        helper(dtype)

# Repeat tests
def test_repeat():
    for dtype in dtypes:
        helper(dtype)

if __name__ == "__main__":
    test()
    test_repeat()

