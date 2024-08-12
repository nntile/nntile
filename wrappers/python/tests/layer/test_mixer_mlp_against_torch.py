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
# @version 1.1.0

import numpy as np
import pytest
import torch

import nntile
from nntile.layer import MixerMlp
from nntile.torch_models.mlp_mixer import MixerMlp as TorchMixerMlp

# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
          np.float64: nntile.tensor.Tensor_fp64}

config = nntile.starpu.Config(1, 0, 0)
nntile.starpu.init()


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('side', ['L', 'R'])
def test_mixer_mlp_with_torch(side: str, dtype: np.dtype):
    if dtype == np.float32:
        tol = 1e-5
    elif dtype == np.float64:
        tol = 1e-10

    # Describe single-tile tensor, located at node 0
    n_patches = 8
    n_channels = 4
    A_shape = [n_patches, 2, n_channels]
    A_traits = nntile.tensor.TensorTraits(A_shape, A_shape)
    mpi_distr = [0]
    next_tag = 0

    # Tensor objects
    A = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    A_grad = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A_grad.next_tag

    # Set initial values of tensors
    rng = np.random.default_rng(42)
    rand_A = rng.standard_normal(A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)

    # Define mlp_mixer layer
    layer, next_tag = MixerMlp.generate_simple(A_moments, side, next_tag)

    rand_W1 = rng.standard_normal(layer.linear_1.w.value.shape)
    np_W1 = np.array(rand_W1, dtype=dtype, order='F')
    layer.linear_1.w.value.from_array(np_W1)

    rand_W2 = rng.standard_normal(layer.linear_2.w.value.shape)
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

    match side:
        case 'L':
            torch_mlp = TorchMixerMlp(side, n_channels)
        case 'R':
            torch_mlp = TorchMixerMlp(side, n_patches)
    torch_mlp.set_weight(np_W1, np_W2)
    torch_mlp.zero_grad()
    torch_output = torch_mlp.forward(torch.from_numpy(np_A))

    torch_loss = 0.5 * torch.sum(torch.square(torch_output))
    torch_loss.backward()

    np_Y = torch_output.detach().numpy().astype(dtype, 'F')
    assert np.linalg.norm(np_Y - np_Y2) / np.linalg.norm(np_Y) <= tol

    param_pairs = zip(layer.parameters, torch_mlp.parameters())
    for i, (p_nntile, p_torch) in enumerate(param_pairs):
        p_nntile_grad_np = np.zeros(p_nntile.grad.shape, dtype, 'F')
        p_nntile.grad.to_array(p_nntile_grad_np)
        match side:
            case 'L':
                abs_error = torch.norm(p_torch.grad -
                                       torch.from_numpy(p_nntile_grad_np).T)
            case 'R':
                abs_error = torch.norm(p_torch.grad -
                                        torch.from_numpy(p_nntile_grad_np))
        rel_error = abs_error / torch.norm(p_torch.grad)
        print(f'Relative error in gradient in layer {i} = {rel_error}')

    A_moments.unregister()
    layer.unregister()
