import nntile
import numpy as np
config = nntile.starpu.Config(1, 0, 0)

def test_tile_array():
    shape = [3, 4]
    basetile = [2, 2]
    traits = nntile.tensor.TensorTraits(shape, basetile)
    mpi_distr = [0] * traits.grid.nelems;
    next_tag = 0
    tensor_fp32 = nntile.tensor.Tensor_fp32(traits, mpi_distr, next_tag)
    next_tag = tensor_fp32.next_tag
    tensor_fp64 = nntile.tensor.Tensor_fp64(traits, mpi_distr, next_tag)
    next_tag = tensor_fp64.next_tag
    src_fp32 = np.array(np.random.randn(*shape), dtype=np.float32, order='F')
    src_fp64 = np.array(np.random.randn(*shape), dtype=np.float64, order='F')
    #tensor_fp32.from_array(src_fp32)
    #tensor_fp64.from_array(src_fp64)
    #dst_fp32 = np.zeros(shape, dtype=np.float32, order='F')
    #dst_fp64 = np.zeros(shape, dtype=np.float64, order='F')
    #tensor_fp32.to_array(dst_fp32)
    #tensor_fp64.to_array(dst_fp64)
    #assert (dst_fp32 == src_fp32).all()
    #assert (dst_fp64 == src_fp64).all()

