import nntile
import numpy as np
config = nntile.starpu.Config(1, 0, 0)

def helper(shape, Tensor, dtype):
    mpi_distr = [0]
    next_tag = 0
    traits = nntile.tensor.TensorTraits(shape, shape)
    tensor = Tensor(traits, mpi_distr, next_tag)
    src = np.array(np.random.randn(*shape), dtype=dtype, order='F')
    dst = np.zeros_like(src)
    tensor.from_array(src)
    tensor.to_array(dst)
    nntile.starpu.wait_for_all()
    tensor.unregister()
    return (dst == src).all()

def test():
    shape = [3, 4]
    assert helper(shape, nntile.tensor.Tensor_fp32, np.float32)
    assert helper(shape, nntile.tensor.Tensor_fp64, np.float64)

def test_repeat():
    shape = [3, 4]
    assert helper(shape, nntile.tensor.Tensor_fp32, np.float32)
    assert helper(shape, nntile.tensor.Tensor_fp64, np.float64)

if __name__ == "__main__":
    test()
    test_repeat()

