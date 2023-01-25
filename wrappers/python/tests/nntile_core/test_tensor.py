# All necesary imports
import nntile
import numpy as np
# Set up StarPU configuration and init it
config = nntile.starpu.Config(1, 0, 0)
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

