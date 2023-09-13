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
    layer.forward_async()

    torch_mlp = TorchMixerMlp('L', n_channels)
    torch_mlp.set_weight(np_W1, np_W2)
    torch_mlp.zero_grad()
    torch_output = torch_mlp.forward(torch.from_numpy(np_A))
    np_Y = np.array(torch_output.detach().numpy(), order="F", dtype=dtype)

    np_Y2 = np.zeros_like(np_Y, order='F')
    layer.y.value.to_array(np_Y2)
    if np.linalg.norm(np_Y-np_Y2)/np.linalg.norm(np_Y) > tol:
        A_moments.unregister()
        layer.unregister()
        return False 

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
    layer.forward_async()

    torch_mlp = TorchMixerMlp('R', n_patches)
    torch_mlp.set_weight(np_W1, np_W2)
    torch_mlp.zero_grad()
    torch_output = torch_mlp.forward(torch.from_numpy(np_A))
    np_Y = np.array(torch_output.detach().numpy(), order="F", dtype=dtype)

    np_Y2 = np.zeros_like(np_Y, order='F')
    layer.y.value.to_array(np_Y2)
    if np.linalg.norm(np_Y-np_Y2)/np.linalg.norm(np_Y) > tol:
        A_moments.unregister()
        layer.unregister()
        return False 

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
