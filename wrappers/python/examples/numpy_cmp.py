from nntile.tensor import TensorTraits, Tensor, TensorOrNone, TensorMoments, \
        copy_async, axpy_async, nrm2_async, prod_async
from nntile.tensor import maxsumexp_async, logsumexp_async

import numpy as np
import scipy.special as spsp
import nntile

if __name__ == "__main__":
    nclasses = 5
    batch_size = 10
    final_layer_output = np.array(np.random.randn(batch_size, nclasses), dtype=np.float32, order="F")
    np_res = spsp.logsumexp(final_layer_output, axis=1)
    # print(np.max(final_layer_output, axis=1))
    config = nntile.starpu.Config(1, 0, 0)
    # Init all NNTile-StarPU codelets
    nntile.starpu.init()
    next_tag = 0
    A_traits = nntile.tensor.TensorTraits(final_layer_output.shape, final_layer_output.shape)
    mpi_distr = [0]
    A = nntile.tensor.Tensor_fp32(A_traits, mpi_distr, next_tag)
    next_tag = A.next_tag
    A.from_array(final_layer_output)
    msexp_traits = nntile.tensor.TensorTraits((2, final_layer_output.shape[0]), 
                                              (2, final_layer_output.shape[0]))
    msexp = nntile.tensor.Tensor_fp32(msexp_traits, mpi_distr, next_tag)
    next_tag = msexp.next_tag
    maxsumexp_async(A, msexp, 1)
    msexp_np = np.zeros((2, final_layer_output.shape[0]), dtype=np.float32, order="F")
    msexp.to_array(msexp_np)
    print(msexp_np[0] + np.log(msexp_np[1]))

    logsexp_traits = nntile.tensor.TensorTraits((final_layer_output.shape[0],), 
                                              (final_layer_output.shape[0],))
    logsexp = nntile.tensor.Tensor_fp32(logsexp_traits, mpi_distr, next_tag)
    logsumexp_async(msexp, logsexp)
    logsexp_np = np.zeros((final_layer_output.shape[0],), dtype=np.float32, order="F")
    logsexp.to_array(logsexp_np)
    print(logsexp_np)

    
    # print(np.log(msexp_np[1]))
    A.unregister()
    msexp.unregister()
    logsexp.unregister()




