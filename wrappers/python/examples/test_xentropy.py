from nntile.tensor import TensorTraits, Tensor, TensorOrNone, TensorMoments, \
        copy_async, axpy_async, nrm2_async, prod_async
from nntile.tensor import maxsumexp_async, logsumexp_async, total_sum_accum_async

import numpy as np
import scipy.special as spsp
import nntile
from nntile.loss.crossentropy import CrossEntropy

def crossentropy_np(final_layer_output, class_labels):
    np_res = spsp.logsumexp(final_layer_output, axis=1)
    xentropy_np = np.sum(np_res - final_layer_output[np.arange(batch_size), true_class_lables])
    return xentropy_np

def grad_crossentropy_np(final_layer_output, class_labels):
    return spsp.softmax(final_layer_output, axis=1)

if __name__ == "__main__":
    nclasses = 5
    batch_size = 7
    final_layer_output = np.array(np.random.randn(batch_size, nclasses), dtype=np.float32, order="F")
    true_class_lables = np.array(np.random.randint(0, nclasses, (batch_size,), dtype=np.int64), order="F")
    print("NumPy x-entropy =", crossentropy_np(final_layer_output, true_class_lables))
    print("Softmax NP =", grad_crossentropy_np(final_layer_output, true_class_lables))
    config = nntile.starpu.Config(1, 0, 0)
    # Init all NNTile-StarPU codelets
    nntile.starpu.init()
    next_tag = 0
    final_layer_output_traits = nntile.tensor.TensorTraits(final_layer_output.shape, final_layer_output.shape)
    mpi_distr = [0]
    final_layer_output_tensor = nntile.tensor.Tensor_fp32(final_layer_output_traits, mpi_distr, next_tag)
    next_tag = final_layer_output_tensor.next_tag
    final_layer_output_tensor.from_array(final_layer_output)

    final_layer_output_grad = nntile.tensor.Tensor_fp32(final_layer_output_traits, mpi_distr, next_tag)
    next_tag = final_layer_output_grad.next_tag
    final_layer_output_tm = nntile.tensor.TensorMoments(final_layer_output_tensor,
                                                        final_layer_output_grad, True)

    class_labels_traits = nntile.tensor.TensorTraits((batch_size,), (batch_size,))
    tensor_class_labels = nntile.tensor.Tensor_int64(class_labels_traits, mpi_distr, next_tag)
    tensor_class_labels.from_array(true_class_lables)
    next_tag = tensor_class_labels.next_tag
    # Create crossentropy loss
    xentropy_loss, next_tag = CrossEntropy.generate_simple(final_layer_output_tm, 
                                                           tensor_class_labels, next_tag)
    xentropy_loss.calc_async()

    nntile_xentropy_np = np.zeros((1,), dtype=np.float32, order="F")
    xentropy_loss.get_val(nntile_xentropy_np)

    nntile_xentropy_grad_np = np.zeros((batch_size, nclasses), dtype=np.float32, order="F")
    xentropy_loss.get_grad(nntile_xentropy_grad_np)


    print("NNtile x-entropy =", nntile_xentropy_np[0])
    print("NNtile softmax =", nntile_xentropy_grad_np)

    xentropy_loss.unregister()
    tensor_class_labels.unregister()
    final_layer_output_tensor.unregister()
    final_layer_output_grad.unregister()




