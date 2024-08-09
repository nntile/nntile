# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_linear.py
# Test for nntile.layer.linear
#
# @version 1.1.0

import numpy as np
import pytest
import torch
import torch.nn as nn

import nntile
import nntile.utils.constructors as nntc
from nntile.layer import Linear

# Define mapping between numpy and nntile types
Tensor = {
    np.float32: nntile.tensor.Tensor_fp32,
    np.float64: nntile.tensor.Tensor_fp64,
}

config = nntile.starpu.Config(1, 0, 0)
nntile.starpu.init()


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('side', ['L', 'R'])
def test_linear(side: str, dtype: np.dtype):
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
    # Set initial values of tensors
    rng = np.random.default_rng(42)
    rand_A = rng.standard_normal(A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)
    # Define linear layer
    layer, next_tag = Linear.generate_simple(
        A_moments, side, nntile.tensor.notrans, 2, [7, 8], [7, 8], next_tag,
        bias=False)
    rand_W = rng.standard_normal(layer.w.value.shape)
    np_W = np.array(rand_W, dtype=dtype, order='F')
    layer.w.value.from_array(np_W)
    nntile.tensor.clear_async(layer.w.grad)

    # Check result of forward pass layer.y.value
    A.from_array(np_A)
    nntile.tensor.clear_async(A_grad)
    layer.forward_async()

    match side:
        case 'L':
            np_Y = np.tensordot(np_A, np_W, 2)
        case 'R':
            np_Y = np.tensordot(np_W, np_A, 2)
    np_Y2 = np.zeros_like(np_Y, order='F')
    layer.y.value.to_array(np_Y2)
    assert np.linalg.norm(np_Y - np_Y2) / np.linalg.norm(np_Y) <= 1e-5

    # Check results of backward pass layer.w.grad and layer.x.grad
    layer.y.grad.from_array(np_Y)
    layer.backward_async()

    match side:
        case 'L':
            np_Z = np.einsum("ijk,ilm->jklm", np_A, np_Y2)
        case 'R':
            np_Z = np.einsum("ijk,lmk->ijlm", np_Y2, np_A)
    np_Z2 = np.zeros_like(np_Z, order='F')
    layer.w.grad.to_array(np_Z2)
    assert np.linalg.norm(np_Z - np_Z2) / np.linalg.norm(np_Z) <= 1e-5

    match side:
        case 'L':
            np_Z3 = np.einsum("ijk,lmjk->ilm", np_Y2, np_W)
        case 'R':
            np_Z3 = np.einsum("ijkl,ijm->klm", np_W, np_Y2)
    np_Z4 = np.zeros_like(np_Z3, order='F')
    layer.x.grad.to_array(np_Z4)
    assert np.linalg.norm(np_Z3 - np_Z4) / np.linalg.norm(np_Z3) < 1e-5

    A_moments.unregister()
    layer.unregister()


@pytest.mark.skip(reason='not implemented')
def test_linear_fp32_fast_fp16():
    dtype = np.float32
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
    # Set initial values of tensors
    rng = np.random.default_rng(42)
    rand_A = rng.standard_normal(A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)
    # Define linear layer
    layer, next_tag = Linear.generate_simple(
        A_moments, 'L', nntile.tensor.notrans, 2, [7, 8], [7, 8], next_tag,
        fp32_fast_fp16=True, bias=False)
    rand_W = rng.standard_normal(layer.w.value.shape)
    np_W = np.array(rand_W, dtype=dtype, order='F')
    layer.w.value.from_array(np_W)
    # Check result of forward pass layer.y.value
    A.from_array(np_A)
    layer.forward_async()
    np_Y = np.tensordot(np_A, np_W, 2)
    np_Y2 = np.zeros_like(np_Y, order='F')
    layer.y.value.to_array(np_Y2)
    assert np.linalg.norm(np_Y - np_Y2) / np.linalg.norm(np_Y) <= 1e-5
    # Check results of backward pass layer.w.grad and layer.x.grad
    layer.y.grad.from_array(np_Y)
    layer.backward_async()
    np_Z = np.einsum("ijk,ilm->jklm", np_A, np_Y2)
    np_Z2 = np.zeros_like(np_Z, order='F')
    layer.w.grad.to_array(np_Z2)
    assert np.linalg.norm(np_Z - np_Z2) / np.linalg.norm(np_Z) <= 1e-5
    np_Z3 = np.einsum("ijk,lmjk->ilm", np_Y2, np_W)
    np_Z4 = np.zeros_like(np_Z3, order='F')
    layer.x.grad.to_array(np_Z4)
    assert np.linalg.norm(np_Z3 - np_Z4) / np.linalg.norm(np_Z3) <= 1e-5
    A_moments.unregister()
    layer.unregister()


# Support for multi-dim bias will be added later.
#   ('L', [20, 10], [10, 5, 7], [5, 7], 1),
#   ('R', [7, 2], [10, 5, 7], [10, 5], 1),
@pytest.mark.parametrize('side,x_shape,w_shape,b_shape,n_contracted_dim', [
    ('L', [20, 10], [10, 5], [5], 1),
    ('L', [20, 10, 5], [10, 5, 7], [7], 2),
    ('L', [20, 10, 5], [5, 7], [7], 1),
    ('R', [5, 3], [10, 5], [10], 1),
    ('R', [5, 7, 2], [10, 5, 7], [10], 2),
    ('R', [7, 3, 10], [5, 7], [5], 1),
])
def test_linear_with_torch(side: str, x_shape, w_shape, b_shape,
                           n_contracted_dim):
    """Compare :py:class:`nntile.layer.Linear` and :py:class:`torch.nn.Linear`.

    There are two cases depending on `side` parameter.

        y = x @ w + b (left)
        y = w @ x + b (right)

    Different shapes of b mean the axis of bias addition.
    """
    w_torch = torch.randn(w_shape, requires_grad=True)
    b_torch = torch.randn(b_shape, requires_grad=True)
    x_torch = torch.randn(x_shape, requires_grad=True)

    match side:
        case 'L':
            y_torch = (torch.tensordot(x_torch, w_torch, n_contracted_dim) +
                       b_torch.view(1, *b_torch.shape))
        case 'R':
            y_torch = torch.tensordot(w_torch, x_torch, n_contracted_dim)
            if len(y_torch.shape) - len(b_torch.shape) == 1:
                y_torch += b_torch.view(*b_torch.shape, 1)
            elif len(y_torch.shape) - len(b_torch.shape) == 2:
                y_torch += b_torch.view(*b_torch.shape, 1, 1)

    loss_torch = y_torch.sum()
    loss_torch.backward()

    A_traits = nntile.tensor.TensorTraits(x_torch.shape, x_torch.shape)
    mpi_distr = [0]
    next_tag = 0
    # Tensor objects
    A = Tensor[np.float32](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    A_grad = Tensor[np.float32](A_traits, mpi_distr, next_tag)
    next_tag = A_grad.next_tag
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)
    # Define linear layer
    match side:
        case 'L':
            layer, next_tag = Linear.generate_simple(
                A_moments, 'L', nntile.tensor.notrans, n_contracted_dim,
                [*w_shape[n_contracted_dim:]], [*w_shape[n_contracted_dim:]],
                next_tag, bias=True)
        case 'R':
            layer, next_tag = Linear.generate_simple(
                A_moments, 'R', nntile.tensor.notrans, n_contracted_dim,
                [*w_shape[:-n_contracted_dim]], [*w_shape[:-n_contracted_dim]],
                next_tag, bias=True)

    np_W = np.array(w_torch.detach().numpy(), dtype=np.float32, order='F')
    layer.w.value.from_array(np_W)
    np_b = np.array(b_torch.detach().numpy(), dtype=np.float32, order='F')
    layer.b.value.from_array(np_b)
    nntile.tensor.clear_async(layer.w.grad)
    nntile.tensor.clear_async(layer.b.grad)
    # Check result of forward pass layer.y.value
    np_A = np.array(x_torch.detach().numpy(), dtype=np.float32, order='F')
    A.from_array(np_A)
    nntile.tensor.clear_async(A_grad)
    layer.forward_async()
    layer.y.grad.from_array(np.ones(y_torch.shape, np.float32, 'F'))
    layer.backward_async()

    nntile_res = np.zeros(y_torch.shape, dtype=np.float32, order="F")
    layer.y.value.to_array(nntile_res)
    output_rel_error = (np.linalg.norm(nntile_res - y_torch.detach().numpy()) /
                        np.linalg.norm(y_torch.detach().numpy()))
    assert output_rel_error <= 1e-5

    nntile_w_grad = np.zeros(w_torch.shape, dtype=np.float32, order="F")
    layer.w.grad.to_array(nntile_w_grad)
    w_grad_rel_error = (
        np.linalg.norm(nntile_w_grad - w_torch.grad.detach().numpy()) /
        np.linalg.norm(w_torch.grad.detach().numpy()))
    assert w_grad_rel_error <= 1e-5

    nntile_b_grad = np.zeros(b_torch.shape, dtype=np.float32, order="F")
    layer.b.grad.to_array(nntile_b_grad)

    b_grad_rel_error = (
        np.linalg.norm(nntile_b_grad - b_torch.grad.detach().numpy()) /
        np.linalg.norm(b_torch.grad.detach().numpy()))
    assert b_grad_rel_error <= 1e-5

    nntile_x_grad = np.zeros(x_torch.shape, dtype=np.float32, order="F")
    A_moments.grad.to_array(nntile_x_grad)
    x_grad_rel_error = (
        np.linalg.norm(nntile_x_grad - x_torch.grad.detach().numpy()) /
        np.linalg.norm(x_torch.grad.detach().numpy()))
    assert x_grad_rel_error <= 1e-5

    A_moments.unregister()
    layer.unregister()


@pytest.mark.parametrize('x_shape,w_shape', [
    ([64, 100], [100, 10]),
    ([64, 128, 100], [100, 20]),
])
def test_linear_with_torch_init(x_shape, w_shape):
    """Similar to `test_linear_with_torch` but weights are initialized with
    :py:class:`torch.nn.Linear`.
    """
    linear_layer = nn.Linear(*w_shape)
    x_torch = torch.randn(x_shape, requires_grad=True)
    torch_output = linear_layer(x_torch)
    loss = torch.sum(torch_output)
    loss.backward()

    A_traits = nntile.tensor.TensorTraits(x_torch.shape, x_torch.shape)
    mpi_distr = [0]
    next_tag = 0
    # Tensor objects
    A = Tensor[np.float32](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    A_grad = Tensor[np.float32](A_traits, mpi_distr, next_tag)
    next_tag = A_grad.next_tag
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)
    # Define linear layer
    layer, next_tag = Linear.generate_simple(
        A_moments, 'L', nntile.tensor.notrans, 1, [w_shape[1]], [w_shape[1]],
        next_tag, bias=True)

    np_W = linear_layer.weight.detach().numpy().astype(np.float32, 'F')
    layer.w.value.from_array(np_W.T)
    np_b = linear_layer.bias.detach().numpy().astype(np.float32, 'F')
    layer.b.value.from_array(np_b)
    nntile.tensor.clear_async(layer.w.grad)
    nntile.tensor.clear_async(layer.b.grad)
    # Check result of forward pass layer.y.value
    np_A = np.array(x_torch.detach().numpy(), dtype=np.float32, order='F')
    A.from_array(np_A)
    nntile.tensor.clear_async(A_grad)
    layer.forward_async()
    layer.y.grad.from_array(np.ones(torch_output.shape, np.float32, 'F'))
    layer.backward_async()

    nntile_res = np.zeros(torch_output.shape, dtype=np.float32, order="F")
    layer.y.value.to_array(nntile_res)
    output_rel_error = (
        np.linalg.norm(nntile_res - torch_output.detach().numpy()) /
        np.linalg.norm(torch_output.detach().numpy()))
    assert output_rel_error <= 1e-5

    nntile_w_grad = np.zeros(linear_layer.weight.T.shape, np.float32, 'F')
    layer.w.grad.to_array(nntile_w_grad)
    w_grad_rel_error = (
        np.linalg.norm(nntile_w_grad.T -
                       linear_layer.weight.grad.detach().numpy()) /
        np.linalg.norm(linear_layer.weight.grad.detach().numpy()))
    assert w_grad_rel_error <= 1e-5

    nntile_b_grad = np.zeros(linear_layer.bias.shape, np.float32, 'F')
    layer.b.grad.to_array(nntile_b_grad)
    b_grad_rel_error = (
        np.linalg.norm(nntile_b_grad - linear_layer.bias.grad.detach().numpy())
        / np.linalg.norm(linear_layer.bias.grad.detach().numpy()))
    assert b_grad_rel_error <= 1e-5

    nntile_x_grad = np.zeros(x_torch.shape, dtype=np.float32, order="F")
    A_moments.grad.to_array(nntile_x_grad)
    x_grad_rel_error = (
        np.linalg.norm(nntile_x_grad - x_torch.grad.detach().numpy()) /
        np.linalg.norm(x_torch.grad.detach().numpy()))
    assert x_grad_rel_error <= 1e-5

    A_moments.unregister()
    layer.unregister()


@pytest.mark.parametrize(
    "x_shape,w_shape",
    [
        ([64, 100], [100, 10]),
        ([64, 128, 100], [100, 20]),
    ],
)
def test_dynamic(numpy_rng, x_shape, w_shape):
    """Similar to `test_linear_with_torch` but weights are initialized with
    :py:class:`torch.nn.Linear`.
    """
    # build layer from torch
    linear_layer = nn.Linear(*w_shape)
    x_nntile_tm_for_build = nntile.tensor.TensorMoments(
        nntc.zeros(x_shape[::-1], dtype=nntile.tensor.Tensor_fp32), None, False
    )
    next_tag = 0
    layer, next_tag = Linear.from_torch(
        linear_layer, x_nntile_tm_for_build, w_shape[1], False, next_tag
    )

    # generate input
    x_np = np.asfortranarray(numpy_rng.random(x_shape, dtype=np.float32))

    # Test for same size tensor
    x_torch = torch.Tensor(x_np)
    torch_output = linear_layer(x_torch)

    x_nntile_tm = nntile.tensor.TensorMoments(
        nntc.from_array(np.transpose(x_np)), None, False
    )
    nntile_res_tm = layer.forward_dynamic(x_nntile_tm)
    nntile_res = np.transpose(nntc.to_numpy(nntile_res_tm.value))

    output_rel_error = np.linalg.norm(
        nntile_res - torch_output.detach().numpy()
    ) / np.linalg.norm(torch_output.detach().numpy())
    assert output_rel_error <= 1e-5

    # Test for half size tensor
    torch_output_half = linear_layer(x_torch[: x_torch.shape[0] // 2, :])
    x_nntile_half_tm = nntile.tensor.TensorMoments(
        nntc.from_array(np.transpose(x_np[: x_np.shape[0] // 2, :])),
        None,
        False,
    )
    nntile_res_half_tm = layer.forward_dynamic(x_nntile_half_tm)
    nntile_half_res = np.transpose(nntc.to_numpy(nntile_res_half_tm.value))

    output_rel_error = np.linalg.norm(
        nntile_half_res - torch_output_half.detach().numpy()
    ) / np.linalg.norm(torch_output_half.detach().numpy())
    assert output_rel_error <= 1e-5

    x_nntile_tm_for_build.value.unregister()
    x_nntile_tm.value.unregister()
    layer.unregister()


@pytest.mark.parametrize('side,x_shape,w_shape,b_shape,n_contracted_dim', [
    ('L', [20, 10], [10, 5], [5], 1),
    ('L', [20, 10, 5], [10, 5, 7], [7], 2),
    ('L', [20, 10, 5], [5, 7], [7], 1),
    ('R', [5, 3], [10, 5], [10], 1),
    ('R', [5, 7, 2], [10, 5, 7], [10], 2),
    ('R', [7, 3, 10], [5, 7], [5], 1),
])
def test_linear_flops(side: str, x_shape, w_shape, b_shape,
                           n_contracted_dim):
    """Compare flops counting in :py:class:`nntile.layer.Linear`
    and analytical formulas
    """

    A_traits = nntile.tensor.TensorTraits(x_shape, x_shape)
    mpi_distr = [0]
    next_tag = 0
    # Tensor objects
    A = Tensor[np.float32](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    A_grad = Tensor[np.float32](A_traits, mpi_distr, next_tag)
    next_tag = A_grad.next_tag
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)
    # Define linear layer
    match side:
        case 'L':
            layer, next_tag = Linear.generate_simple(
                A_moments, 'L', nntile.tensor.notrans, n_contracted_dim,
                [*w_shape[n_contracted_dim:]], [*w_shape[n_contracted_dim:]],
                next_tag, bias=True)
        case 'R':
            layer, next_tag = Linear.generate_simple(
                A_moments, 'R', nntile.tensor.notrans, n_contracted_dim,
                [*w_shape[:-n_contracted_dim]], [*w_shape[:-n_contracted_dim]],
                next_tag, bias=True)
    match side:
        case 'L':
            analytical_fwd_flops = (2 * np.prod(x_shape) *
                                    np.prod(w_shape[n_contracted_dim:]))
        case 'R':
            analytical_fwd_flops = (2 * np.prod(x_shape) *
                                    np.prod(w_shape[:-n_contracted_dim:]))

    assert analytical_fwd_flops == layer.get_forward_flops()

    analytical_bwd_flops = 2 * analytical_fwd_flops
    assert analytical_bwd_flops == layer.get_backward_flops()

    A_moments.unregister()
    layer.unregister()
