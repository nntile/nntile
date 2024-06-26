# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/loss/test_xentropy.py
# Test for nntile.loss.CrossEntropy
#
# @version 1.0.0

# All necesary imports
import nntile
import numpy as np
import numpy as np
import scipy.special as spsp

# Set up StarPU configuration and init it
config = nntile.starpu.Config(1, 0, 0)
# Init all NNTile-StarPU codelets
nntile.starpu.init()
# Define list of tested types
dtypes = [np.float32, np.float64]
# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
        np.float64: nntile.tensor.Tensor_fp64}
# Get multiprecision loss function
cross_entropy = nntile.loss.CrossEntropy

def crossentropy_np(final_layer_output, class_labels):
    np_res = spsp.logsumexp(final_layer_output, axis=1)
    xentropy_np = np.sum(np_res - final_layer_output[np.arange(final_layer_output.shape[0]),
                                                     class_labels])
    return xentropy_np

def grad_crossentropy_np(final_layer_output, class_labels):
    softmax = spsp.softmax(final_layer_output, axis=1)
    softmax[np.arange(class_labels.shape[0]), class_labels] -= 1
    return softmax

# Helper function returns bool value true if test passes
def helper(dtype: np.dtype):
    # Describe single-tile tensor, located at node 0
    nclasses = 5
    batch_size = 7
    final_layer_output = np.array(np.random.randn(batch_size, nclasses), dtype=dtype, order="F")
    true_class_lables = np.array(np.random.randint(0, nclasses, (batch_size,), dtype=np.int64), order="F")

    np_xentropy = crossentropy_np(final_layer_output, true_class_lables)
    np_xentropy_grad = grad_crossentropy_np(final_layer_output, true_class_lables)

    next_tag = 0
    final_layer_output_traits = nntile.tensor.TensorTraits( \
            [nclasses, batch_size], [nclasses, batch_size])
    mpi_distr = [0]
    final_layer_output_tensor = Tensor[dtype](final_layer_output_traits, mpi_distr, next_tag)
    next_tag = final_layer_output_tensor.next_tag
    final_layer_output_tensor.from_array(final_layer_output.T)

    final_layer_output_grad = Tensor[dtype](final_layer_output_traits, mpi_distr, next_tag)
    next_tag = final_layer_output_grad.next_tag
    final_layer_output_tm = nntile.tensor.TensorMoments(final_layer_output_tensor,
                                                        final_layer_output_grad, True)

    # Create crossentropy loss
    xentropy_loss, next_tag = cross_entropy.generate_simple( \
            final_layer_output_tm, next_tag)
    xentropy_loss.y.from_array(true_class_lables)
    xentropy_loss.calc_async()

    nntile_xentropy_np = np.zeros((1,), dtype=dtype, order="F")
    xentropy_loss.get_val(nntile_xentropy_np)

    nntile_xentropy_grad_np = np.zeros((nclasses, batch_size), dtype=dtype, order="F")
    xentropy_loss.get_grad(nntile_xentropy_grad_np)

    xentropy_loss.unregister()
    final_layer_output_tensor.unregister()
    final_layer_output_grad.unregister()
    if dtype == np.float32:
        tol = 1e-5
    elif dtype == np.float64:
        tol = 1e-10
    if np.max(np.abs(np_xentropy_grad - nntile_xentropy_grad_np.T)) > tol:
        return False

    if np.abs(nntile_xentropy_np[0] - np_xentropy) > tol:
        return False

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
