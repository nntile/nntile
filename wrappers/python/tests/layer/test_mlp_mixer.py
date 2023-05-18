# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# Test for nntile.layer.mlp_mixer
#
# @version 1.0.0
# @author Gleb Karpov
# @date 2023-04-23

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
# Get mlp-mixer layer
MlpMixer = nntile.layer.MlpMixer

# Helper function returns bool value true if test passes
def helper(dtype: np.dtype):
    if dtype == np.float32:
        tol = 1e-5
    elif dtype == np.float64:
        tol = 1e-10
    # Describe single-tile tensor, located at node 0
    A_shape = [1, 8, 4]
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

    # Define mlp_mixer layer
    layer, next_tag = MlpMixer.generate_simple_mpiroot(A_moments, 'L',
            nntile.tensor.notrans, 1, [16], [16], next_tag)
    
    rand_W1 = np.random.randn(*layer.w1.value.shape)
    np_W1 = np.array(rand_W1, dtype=dtype, order='F')
    layer.w1.value.from_array(np_W1)

    rand_W2 = np.random.randn(*layer.w2.value.shape)
    np_W2 = np.array(rand_W2, dtype=dtype, order='F')
    layer.w2.value.from_array(np_W2)

    A.from_array(np_A)
    layer.forward_async()

    np_Y_interm = np.tensordot(np_A, np_W1, 1)
    torch_output = F.gelu(torch.from_numpy(np_Y_interm))
    np_Y_interm2 = np.array(torch_output.numpy(), order="F", dtype=dtype)
    np_Y = np.tensordot(np_Y_interm2, np_W2, 1)

    np_Y2 = np.zeros_like(np_Y, order='F')
    layer.y.value.to_array(np_Y2)
    if np.linalg.norm(np_Y-np_Y2)/np.linalg.norm(np_Y) > 1e-5:
        A_moments.unregister()
        layer.unregister()
        return False 

    A_moments.unregister()
    layer.unregister()
    print("Finish checking")
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

