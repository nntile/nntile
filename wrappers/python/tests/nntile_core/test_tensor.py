# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor.py
# Test for Tensor<T> Python wrapper
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-12-18

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

# Helper function returns bool value true if test passes
def helper(dtype):
    shape = [3, 4]
    mpi_distr = [0]
    next_tag = 0
    traits = nntile.tensor.TensorTraits(shape, shape)
    tensor = Tensor[dtype](traits, mpi_distr, next_tag)
    src = np.array(np.random.randn(*shape), dtype=dtype, order='F')
    dst = np.zeros_like(src)
    tensor.from_array(src)
    tensor.to_array(dst)
    nntile.starpu.wait_for_all()
    tensor.unregister()
    return (dst == src).all()

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

