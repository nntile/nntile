# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_mixer_mlp_against_torch.py
# Test for MixerMLP layer
#
# @version 1.0.0

import nntile
import numpy as np
import torch.nn.functional as F
import torch
from nntile.torch_models.mlp_mixer import MixerMlp as TorchMixerMlp


# Set up StarPU configuration and init it
config = nntile.starpu.Config(1, 0, 0)
# Init all NNTile-StarPU codelets
nntile.starpu.init()
# Define list of tested types
dtypes = [np.float32, np.float64]
# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
        np.float64: nntile.tensor.Tensor_fp64}

# Get MixerMlp layer from nntile
MixerMlp = nntile.layer.MixerMlp

# Helper function returns bool value true if test passes
def helper_l(dtype: np.dtype):
    if dtype == np.float32:
        tol = 1e-5
    elif dtype == np.float64:
        tol = 1e-10
    # Describe single-tile tensor, located at node 0
    A_shape = [8, 2, 4]
    A_traits = nntile.tensor.TensorTraits(A_shape, A_shape)
    mpi_distr = [0]
    next_tag = 0

    n_channels = A_shape[2]

    # Tensor objects
    A = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    A_grad = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A_grad.next_tag

    # Set initial values of tensors
    rand_A = np.random.randn(*A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)

    # Define mlp_mixer layer
    layer, next_tag = MixerMlp.generate_simple(A_moments, 'L', next_tag)

    rand_W1 = np.random.randn(*layer.linear_1.w.value.shape)
    np_W1 = np.array(rand_W1, dtype=dtype, order='F')
    layer.linear_1.w.value.from_array(np_W1)

    rand_W2 = np.random.randn(*layer.linear_2.w.value.shape)
    np_W2 = np.array(rand_W2, dtype=dtype, order='F')
    layer.linear_2.w.value.from_array(np_W2)

    A.from_array(np_A)

    layer.clear_gradients()
    layer.forward_async()
    nntile.starpu.wait_for_all()

    np_Y2 = np.zeros(layer.y.value.shape, dtype=dtype, order="F")
    layer.y.value.to_array(np_Y2)
    fro_loss, next_tag = nntile.loss.Frob.generate_simple(layer.y, next_tag)
    np_zero = np.zeros(layer.y.value.shape, dtype=dtype, order="F")
    fro_loss.y.from_array(np_zero)
    fro_loss.calc_async()

    layer.backward_async()
    nntile.starpu.wait_for_all()

    torch_mlp = TorchMixerMlp('L', n_channels)
    torch_mlp.set_weight(np_W1, np_W2)
    torch_mlp.zero_grad()
    torch_output = torch_mlp.forward(torch.from_numpy(np_A))

    torch_loss = 0.5 * torch.sum(torch.square(torch_output))
    torch_loss.backward()

    np_Y = np.array(torch_output.detach().numpy(), order="F", dtype=dtype)

    # np_Y2 = np.zeros_like(np_Y, order='F')
    # layer.y.value.to_array(np_Y2)
    if np.linalg.norm(np_Y-np_Y2)/np.linalg.norm(np_Y) > tol:
        A_moments.unregister()
        layer.unregister()
        return False

    for i, (p_nntile, p_torch) in enumerate(zip(layer.parameters, torch_mlp.parameters())):
        p_nntile_grad_np = np.zeros(p_nntile.grad.shape, order="F", dtype=dtype)
        p_nntile.grad.to_array(p_nntile_grad_np)
        # print(p_nntile_grad_np)
        rel_error = torch.norm(p_torch.grad - torch.from_numpy(p_nntile_grad_np).T) / torch.norm(p_torch.grad)
        print("Relative error in gradient in layer {} = {}".format(i, rel_error.item()))

    A_moments.unregister()
    layer.unregister()
    print("helper_l test done")
    assert True


def helper_r(dtype: np.dtype):
    if dtype == np.float32:
        tol = 1e-5
    elif dtype == np.float64:
        tol = 1e-10
    # Describe single-tile tensor, located at node 0
    A_shape = [8, 2, 4]
    A_traits = nntile.tensor.TensorTraits(A_shape, A_shape)
    mpi_distr = [0]
    next_tag = 0

    n_patches = A_shape[0]

    # Tensor objects
    A = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    A_grad = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A_grad.next_tag

    # Set initial values of tensors
    rand_A = np.random.randn(*A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)

    # Define mlp_mixer layer
    layer, next_tag = MixerMlp.generate_simple(A_moments, 'R', next_tag)

    rand_W1 = np.random.randn(*layer.linear_1.w.value.shape)
    np_W1 = np.array(rand_W1, dtype=dtype, order='F')
    layer.linear_1.w.value.from_array(np_W1)

    rand_W2 = np.random.randn(*layer.linear_2.w.value.shape)
    np_W2 = np.array(rand_W2, dtype=dtype, order='F')
    layer.linear_2.w.value.from_array(np_W2)

    A.from_array(np_A)

    layer.clear_gradients()
    layer.forward_async()
    nntile.starpu.wait_for_all()

    np_Y2 = np.zeros(layer.y.value.shape, dtype=dtype, order="F")
    layer.y.value.to_array(np_Y2)
    fro_loss, next_tag = nntile.loss.Frob.generate_simple(layer.y, next_tag)
    np_zero = np.zeros(layer.y.value.shape, dtype=dtype, order="F")
    fro_loss.y.from_array(np_zero)
    fro_loss.calc_async()

    layer.backward_async()
    nntile.starpu.wait_for_all()

    torch_mlp = TorchMixerMlp('R', n_patches)
    torch_mlp.set_weight(np_W1, np_W2)
    torch_mlp.zero_grad()
    torch_output = torch_mlp.forward(torch.from_numpy(np_A))

    torch_loss = 0.5 * torch.sum(torch.square(torch_output))
    torch_loss.backward()

    np_Y = np.array(torch_output.detach().numpy(), order="F", dtype=dtype)

    # np_Y2 = np.zeros_like(np_Y, order='F')
    # layer.y.value.to_array(np_Y2)
    if np.linalg.norm(np_Y-np_Y2)/np.linalg.norm(np_Y) > tol:
        A_moments.unregister()
        layer.unregister()
        return False

    for i, (p_nntile, p_torch) in enumerate(zip(layer.parameters, torch_mlp.parameters())):
        p_nntile_grad_np = np.zeros(p_nntile.grad.shape, order="F", dtype=dtype)
        p_nntile.grad.to_array(p_nntile_grad_np)
        # print(p_nntile_grad_np)
        rel_error = torch.norm(p_torch.grad - torch.from_numpy(p_nntile_grad_np)) / torch.norm(p_torch.grad)
        print("Relative error in gradient in layer {} = {}".format(i, rel_error.item()))

    A_moments.unregister()
    layer.unregister()
    print("helper_r test done")
    assert True

# Test runner for different precisions
def test():
    for dtype in dtypes:
        helper_l(dtype)
        helper_r(dtype)


# Repeat tests
def test_repeat():
    for dtype in dtypes:
        helper_l(dtype)
        helper_r(dtype)


if __name__ == "__main__":
    test()
