# All necesary imports
import nntile
import numpy as np
from math import e
# Set up StarPU configuration and init it
config = nntile.starpu.Config(1, 0, 0)
# Init all NNTile-StarPU codelets
nntile.starpu.init()
# Define list of tested types
dtypes = [np.float32, np.float64]
# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
        np.float64: nntile.tensor.Tensor_fp64}
# Define mapping between tested function and numpy type
softmax = {np.float32: nntile.tensor.softmax_fp32,
        np.float64: nntile.tensor.softmax_fp64}

# Helper function returns bool value true if test passes
def helper(dtype):
    # Describe single-tile tensor, located at node 0
    A_shape = [2, 3, 4]
    B_shape = []
    ndim = len(A_shape)
    for i in range(ndim):
        B_shape.append([2]+A_shape[:i]+A_shape[i+1:])
    mpi_distr = [0]
    next_tag = 0
    A_traits = nntile.tensor.TensorTraits(A_shape, A_shape)
    B_traits = []
    for i in range(ndim):
        B_traits.append(nntile.tensor.TensorTraits(B_shape[i], B_shape[i]))
    # Tensor objects
    A = Tensor[dtype](A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    B = []
    for i in range(ndim):
        B.append(Tensor[dtype](B_traits[i], mpi_distr, next_tag))
        next_tag = B[-1].next_tag
    # Set initial values of tensors
    rand_A = np.random.randn(*A_shape)
    np_A = np.array(rand_A, dtype=dtype, order='F')
    np_A2 = np.zeros_like(np_A)
    np_B = []
    for i in range(ndim):
        rand_B = np.random.randn(*B_shape[i])
        np_B.append(np.array(rand_B, dtype=dtype, order='F'))
        B[i].from_array(np_B[-1])
    # Check result along each axis
    for i in range(ndim):
        A.from_array(np_A)
        softmax[dtype](B[i], A, i)
        B[i].unregister()
        A.to_array(np_A2)
        np_B_max = np.expand_dims(np_B[i][0, ...], axis=i)
        np_B_max = np.repeat(np_B_max, A_shape[i], axis=i)
        np_B_sumexp = np.expand_dims(np_B[i][1, ...], axis=i)
        np_B_sumexp = np.repeat(np_B_sumexp, A_shape[i], axis=i)
        np_C = (e**(np_A-np_B_max)) / np_B_sumexp
        nntile.starpu.wait_for_all()
        if not np.allclose(np_C, np_A2):
            return False
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

