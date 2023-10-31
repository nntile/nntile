# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/test_linear.py
# Test for nntile.layer.linear
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @author Aleksandr Katrutsa
# @date 2023-09-02

# All necesary imports
import nntile
import numpy as np
import torch
import torch.nn as nn

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
Linear = nntile.layer.Linear

# Helper function returns bool value true if test passes
def helper_l(dtype: np.dtype):
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
    # Set initial values of tensors
    rand_A = np.random.randn(*A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)
    # Define linear layer
    layer, next_tag = Linear.generate_simple(A_moments, 'L',
            nntile.tensor.notrans, 2, [7, 8], [7, 8], next_tag, bias=False)
    rand_W = np.random.randn(*layer.w.value.shape)
    np_W = np.array(rand_W, dtype=dtype, order='F')
    layer.w.value.from_array(np_W)
    nntile.tensor.clear_async(layer.w.grad)
    # Check result of forward pass layer.y.value
    A.from_array(np_A)
    nntile.tensor.clear_async(A_grad)
    layer.forward_async()
    np_Y = np.tensordot(np_A, np_W, 2)
    np_Y2 = np.zeros_like(np_Y, order='F')
    layer.y.value.to_array(np_Y2)
    if np.linalg.norm(np_Y-np_Y2)/np.linalg.norm(np_Y) > 1e-5:
        A_moments.unregister()
        layer.unregister()
        return False
    # Check results of backward pass layer.w.grad and layer.x.grad
    layer.y.grad.from_array(np_Y)
    layer.backward_async()
    np_Z = np.einsum("ijk,ilm->jklm", np_A, np_Y2)
    np_Z2 = np.zeros_like(np_Z, order='F')
    layer.w.grad.to_array(np_Z2)
    if np.linalg.norm(np_Z-np_Z2)/np.linalg.norm(np_Z) > 1e-5:
        A_moments.unregister()
        layer.unregister()
        return False
    np_Z3 = np.einsum("ijk,lmjk->ilm", np_Y2, np_W)
    np_Z4 = np.zeros_like(np_Z3, order='F')
    layer.x.grad.to_array(np_Z4)
    if np.linalg.norm(np_Z3-np_Z4)/np.linalg.norm(np_Z3) > 1e-5:
        A_moments.unregister()
        layer.unregister()
        return False
    A_moments.unregister()
    layer.unregister()
    return True

# Helper function returns bool value true if test passes
def helper_r(dtype: np.dtype):
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
    # Set initial values of tensors
    rand_A = np.random.randn(*A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)
    # Define linear layer
    layer, next_tag = Linear.generate_simple(A_moments, 'R',
            nntile.tensor.notrans, 2, [7, 8], [7, 8], next_tag, bias=False)
    rand_W = np.random.randn(*layer.w.value.shape)
    np_W = np.array(rand_W, dtype=dtype, order='F')
    layer.w.value.from_array(np_W)
    nntile.tensor.clear_async(layer.w.grad)
    # Check result of forward pass layer.y.value
    A.from_array(np_A)
    nntile.tensor.clear_async(A_grad)
    layer.forward_async()
    np_Y = np.tensordot(np_W, np_A, 2)
    np_Y2 = np.zeros_like(np_Y, order='F')
    layer.y.value.to_array(np_Y2)
    if np.linalg.norm(np_Y-np_Y2)/np.linalg.norm(np_Y) > 1e-5:
        A_moments.unregister()
        layer.unregister()
        return False
    # Check results of backward pass layer.w.grad and layer.x.grad
    layer.y.grad.from_array(np_Y)
    layer.backward_async()
    np_Z = np.einsum("ijk,lmk->ijlm", np_Y2, np_A)
    np_Z2 = np.zeros_like(np_Z, order='F')
    layer.w.grad.to_array(np_Z2)
    if np.linalg.norm(np_Z-np_Z2)/np.linalg.norm(np_Z) > 1e-5:
        A_moments.unregister()
        layer.unregister()
        return False
    np_Z3 = np.einsum("ijkl,ijm->klm", np_W, np_Y2)
    np_Z4 = np.zeros_like(np_Z3, order='F')
    layer.x.grad.to_array(np_Z4)
    if np.linalg.norm(np_Z3-np_Z4)/np.linalg.norm(np_Z3) > 1e-5:
        A_moments.unregister()
        layer.unregister()
        return False
    A_moments.unregister()
    layer.unregister()
    return True

# Helper function returns bool value true if test passes
def helper_l_fp32_fast_fp16():
    dtype = np.float32
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
    # Set initial values of tensors
    rand_A = np.random.randn(*A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    A_moments = nntile.tensor.TensorMoments(A, A_grad, True)
    # Define linear layer
    layer, next_tag = Linear.generate_simple(A_moments, 'L',
            nntile.tensor.notrans, 2, [7, 8], [7, 8], next_tag, \
            fp32_fast_fp16=True, bias=False)
    rand_W = np.random.randn(*layer.w.value.shape)
    np_W = np.array(rand_W, dtype=dtype, order='F')
    layer.w.value.from_array(np_W)
    # Check result of forward pass layer.y.value
    A.from_array(np_A)
    layer.forward_async()
    np_Y = np.tensordot(np_A, np_W, 2)
    np_Y2 = np.zeros_like(np_Y, order='F')
    layer.y.value.to_array(np_Y2)
    if np.linalg.norm(np_Y-np_Y2)/np.linalg.norm(np_Y) > 1e-5:
        A_moments.unregister()
        layer.unregister()
        return False
    # Check results of backward pass layer.w.grad and layer.x.grad
    layer.y.grad.from_array(np_Y)
    layer.backward_async()
    np_Z = np.einsum("ijk,ilm->jklm", np_A, np_Y2)
    np_Z2 = np.zeros_like(np_Z, order='F')
    layer.w.grad.to_array(np_Z2)
    if np.linalg.norm(np_Z-np_Z2)/np.linalg.norm(np_Z) > 1e-5:
        A_moments.unregister()
        layer.unregister()
        return False
    np_Z3 = np.einsum("ijk,lmjk->ilm", np_Y2, np_W)
    np_Z4 = np.zeros_like(np_Z3, order='F')
    layer.x.grad.to_array(np_Z4)
    if np.linalg.norm(np_Z3-np_Z4)/np.linalg.norm(np_Z3) > 1e-5:
        A_moments.unregister()
        layer.unregister()
        return False
    A_moments.unregister()
    layer.unregister()
    return True

def helper_torch_l(x_shape, w_shape, b_shape, n_contracted_dim):
    '''
    y = x @ w + b
    different shapes of b mean the axis of bias addition
    '''
    w_torch = torch.randn(w_shape, requires_grad=True)
    b_torch = torch.randn(b_shape, requires_grad=True)
    x_torch = torch.randn(x_shape, requires_grad=True)
    
    y_torch = torch.tensordot(x_torch, w_torch, n_contracted_dim) \
            + b_torch.view(1, *b_torch.shape)
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
    layer, next_tag = Linear.generate_simple(A_moments, 'L', \
            nntile.tensor.notrans, n_contracted_dim, \
            [*w_shape[n_contracted_dim:]], \
            [*w_shape[n_contracted_dim:]], next_tag, bias=True)
    
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
    layer.y.grad.from_array(np.array(np.ones(y_torch.shape), order="F", \
            dtype=np.float32))
    layer.backward_async()

    nntile_res = np.zeros(y_torch.shape, dtype=np.float32, order="F")
    layer.y.value.to_array(nntile_res)
    output_rel_error = np.linalg.norm(nntile_res - y_torch.detach().numpy()) \
            / np.linalg.norm(y_torch.detach().numpy())
    if output_rel_error > 1e-5:
        print("Output rel error = {}".format(output_rel_error))
        A_moments.unregister()
        layer.unregister()
        assert False
    
    nntile_w_grad = np.zeros(w_torch.shape, dtype=np.float32, order="F")
    layer.w.grad.to_array(nntile_w_grad)
    w_grad_rel_error = np.linalg.norm(nntile_w_grad - \
            w_torch.grad.detach().numpy()) \
            / np.linalg.norm(w_torch.grad.detach().numpy())
    if w_grad_rel_error > 1e-5:
        print("W grad rel error = {}".format(w_grad_rel_error))
        A_moments.unregister()
        layer.unregister()
        assert False

    nntile_b_grad = np.zeros(b_torch.shape, dtype=np.float32, order="F")
    layer.b.grad.to_array(nntile_b_grad)

    b_grad_rel_error = np.linalg.norm(nntile_b_grad - \
            b_torch.grad.detach().numpy()) \
            / np.linalg.norm(b_torch.grad.detach().numpy())
    if b_grad_rel_error > 1e-5:
        print("b grad rel error = {}".format(b_grad_rel_error))
        A_moments.unregister()
        layer.unregister()
        assert False
    
    nntile_x_grad = np.zeros(x_torch.shape, dtype=np.float32, order="F")
    A_moments.grad.to_array(nntile_x_grad)
    x_grad_rel_error = np.linalg.norm(nntile_x_grad - \
            x_torch.grad.detach().numpy()) \
            / np.linalg.norm(x_torch.grad.detach().numpy())
    if x_grad_rel_error > 1e-5:
        print("x grad rel error = {}".format(x_grad_rel_error))
        A_moments.unregister()
        layer.unregister()
        assert False

    A_moments.unregister()
    layer.unregister()

    return True

def helper_torch_r(x_shape, w_shape, b_shape, n_contracted_dim):
    '''
    y = w @ x + b
    different shapes of b mean the axis of bias addition
    '''
    w_torch = torch.randn(w_shape, requires_grad=True)
    b_torch = torch.randn(b_shape, requires_grad=True)
    x_torch = torch.randn(x_shape, requires_grad=True)

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
    layer, next_tag = Linear.generate_simple(A_moments, 'R', \
            nntile.tensor.notrans, n_contracted_dim, \
            [*w_shape[:-n_contracted_dim]], \
            [*w_shape[:-n_contracted_dim]], next_tag, bias=True)
    
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
    layer.y.grad.from_array(np.array(np.ones(y_torch.shape), order="F", \
            dtype=np.float32))
    layer.backward_async()

    nntile_res = np.zeros(y_torch.shape, dtype=np.float32, order="F")
    layer.y.value.to_array(nntile_res)
    output_rel_error = np.linalg.norm(nntile_res - y_torch.detach().numpy()) \
            / np.linalg.norm(y_torch.detach().numpy())
    if output_rel_error > 1e-5:
        print("Output rel error = {}".format(output_rel_error))
        A_moments.unregister()
        layer.unregister()
        assert False
    
    nntile_w_grad = np.zeros(w_torch.shape, dtype=np.float32, order="F")
    layer.w.grad.to_array(nntile_w_grad)
    w_grad_rel_error = np.linalg.norm(nntile_w_grad - \
            w_torch.grad.detach().numpy()) \
            / np.linalg.norm(w_torch.grad.detach().numpy())
    if w_grad_rel_error > 1e-5:
        print("W grad rel error = {}".format(w_grad_rel_error))
        A_moments.unregister()
        layer.unregister()
        assert False

    nntile_b_grad = np.zeros(b_torch.shape, dtype=np.float32, order="F")
    layer.b.grad.to_array(nntile_b_grad)

    b_grad_rel_error = np.linalg.norm(nntile_b_grad - \
            b_torch.grad.detach().numpy()) \
            / np.linalg.norm(b_torch.grad.detach().numpy())
    if b_grad_rel_error > 1e-5:
        print("b grad rel error = {}".format(b_grad_rel_error))
        A_moments.unregister()
        layer.unregister()
        assert False
    
    nntile_x_grad = np.zeros(x_torch.shape, dtype=np.float32, order="F")
    A_moments.grad.to_array(nntile_x_grad)
    x_grad_rel_error = np.linalg.norm(nntile_x_grad - \
            x_torch.grad.detach().numpy()) \
            / np.linalg.norm(x_torch.grad.detach().numpy())
    if x_grad_rel_error > 1e-5:
        print("x grad rel error = {}".format(x_grad_rel_error))
        A_moments.unregister()
        layer.unregister()
        assert False

    A_moments.unregister()
    layer.unregister()

    return True


def helper_torch_linear(x_shape, w_shape):
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
    layer, next_tag = Linear.generate_simple(A_moments, 'L', \
            nntile.tensor.notrans, 1, [w_shape[1]], [w_shape[1]], next_tag, \
            bias=True)
    
    np_W = np.array(linear_layer.weight.detach().numpy(), dtype=np.float32, \
            order='F')
    layer.w.value.from_array(np_W.T)
    np_b = np.array(linear_layer.bias.detach().numpy(), dtype=np.float32, \
            order='F')
    layer.b.value.from_array(np_b)
    nntile.tensor.clear_async(layer.w.grad)
    nntile.tensor.clear_async(layer.b.grad)
    # Check result of forward pass layer.y.value
    np_A = np.array(x_torch.detach().numpy(), dtype=np.float32, order='F')
    A.from_array(np_A)
    nntile.tensor.clear_async(A_grad)
    layer.forward_async()
    layer.y.grad.from_array(np.array(np.ones(torch_output.shape), order="F", \
            dtype=np.float32))
    layer.backward_async()

    nntile_res = np.zeros(torch_output.shape, dtype=np.float32, order="F")
    layer.y.value.to_array(nntile_res)
    output_rel_error = np.linalg.norm(nntile_res - \
            torch_output.detach().numpy()) \
            / np.linalg.norm(torch_output.detach().numpy())
    if output_rel_error > 1e-5:
        print("Output rel error = {}".format(output_rel_error))
        A_moments.unregister()
        layer.unregister()
        return False
    
    nntile_w_grad = np.zeros(linear_layer.weight.T.shape, dtype=np.float32, \
            order="F")
    layer.w.grad.to_array(nntile_w_grad)
    w_grad_rel_error = np.linalg.norm(nntile_w_grad.T - \
            linear_layer.weight.grad.detach().numpy()) \
            / np.linalg.norm(linear_layer.weight.grad.detach().numpy())
    if w_grad_rel_error > 1e-5:
        print("W grad rel error = {}".format(w_grad_rel_error))
        A_moments.unregister()
        layer.unregister()
        return False

    nntile_b_grad = np.zeros(linear_layer.bias.shape, dtype=np.float32, \
            order="F")
    layer.b.grad.to_array(nntile_b_grad)
    b_grad_rel_error = np.linalg.norm(nntile_b_grad - \
            linear_layer.bias.grad.detach().numpy()) \
            / np.linalg.norm(linear_layer.bias.grad.detach().numpy())
    if b_grad_rel_error > 1e-5:
        print("b grad rel error = {}".format(b_grad_rel_error))
        A_moments.unregister()
        layer.unregister()
        return False
    
    nntile_x_grad = np.zeros(x_torch.shape, dtype=np.float32, order="F")
    A_moments.grad.to_array(nntile_x_grad)
    x_grad_rel_error = np.linalg.norm(nntile_x_grad - \
            x_torch.grad.detach().numpy()) \
            / np.linalg.norm(x_torch.grad.detach().numpy())
    if x_grad_rel_error > 1e-5:
        print("x grad rel error = {}".format(x_grad_rel_error))
        A_moments.unregister()
        layer.unregister()
        return False

    A_moments.unregister()
    layer.unregister()

    return True


# Test runner for different precisions
def test():
    for dtype in dtypes:
        assert helper_l(dtype)
        assert helper_r(dtype)
    #assert helper_l_fp32_fast_fp16()

# Repeat tests
def test_repeat():
    for dtype in dtypes:
        assert helper_l(dtype)
        assert helper_r(dtype)
    #assert helper_l_fp32_fast_fp16()
    assert helper_torch_l(x_shape=[20, 10],
                          w_shape=[10, 5], b_shape=[5],
                          n_contracted_dim=1)
    # Support for multi-dim bias will be added later
    #assert helper_torch_l(x_shape=[20, 10], 
    #                      w_shape=[10, 5, 7], b_shape=[5, 7],
    #                      n_contracted_dim=1)
    assert helper_torch_l(x_shape=[20, 10, 5],
                          w_shape=[10, 5, 7], b_shape=[7],
                          n_contracted_dim=2)
    assert helper_torch_l(x_shape=[20, 10, 5],
                          w_shape=[5, 7], b_shape=[7],
                          n_contracted_dim=1)
    
    assert helper_torch_r(x_shape=[5, 3], w_shape=[10, 5], 
                          b_shape=[10], n_contracted_dim=1)
    # Support for multi-dim bias will be added later
    #assert helper_torch_r(x_shape=[7, 2], w_shape=[10, 5, 7], 
    #                      b_shape=[10, 5], n_contracted_dim=1)
    assert helper_torch_r(x_shape=[5, 7, 2], w_shape=[10, 5, 7], 
                          b_shape=[10], n_contracted_dim=2)
    assert helper_torch_r(x_shape=[7, 3, 10],
                          w_shape=[5, 7], b_shape=[5],
                          n_contracted_dim=1)


    assert helper_torch_linear(x_shape=[64, 100], w_shape=[100, 10])
    assert helper_torch_linear(x_shape=[64, 128, 100], w_shape=[100, 20])

if __name__ == "__main__":
    test()
    test_repeat()
