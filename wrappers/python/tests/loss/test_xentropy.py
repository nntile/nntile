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
# @version 1.1.0

import numpy as np
import pytest
import scipy.special as spsp

import nntile
from nntile.loss import CrossEntropy as cross_entropy
from nntile.tensor import TensorMoments

config = nntile.starpu.Config(1, 0, 0)
nntile.starpu.init()

# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
          np.float64: nntile.tensor.Tensor_fp64}


def crossentropy_np(final_layer_output, class_labels):
    np_res = spsp.logsumexp(final_layer_output, axis=1)
    np_res -= final_layer_output[np.arange(final_layer_output.shape[0]),
                                 class_labels]
    xentropy_np = np_res.sum()
    return xentropy_np


def grad_crossentropy_np(final_layer_output, class_labels):
    softmax = spsp.softmax(final_layer_output, axis=1)
    softmax[np.arange(class_labels.shape[0]), class_labels] -= 1
    return softmax


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_cross_entropy(dtype: np.dtype):
    """Helper function returns bool value true if test passes."""
    rng = np.random.default_rng(42)

    # Describe single-tile tensor, located at node 0
    nclasses = 5
    batch_size = 7
    final_layer_output = rng \
        .standard_normal((batch_size, nclasses)) \
        .astype(dtype, 'F')
    true_class_lables = rng \
        .integers(0, nclasses, (batch_size,)) \
        .astype(np.int64, 'F')

    np_xentropy = crossentropy_np(final_layer_output, true_class_lables)
    np_xentropy_grad = grad_crossentropy_np(final_layer_output,
                                            true_class_lables)

    next_tag = 0
    final_layer_output_traits = nntile.tensor.TensorTraits(
        [nclasses, batch_size], [nclasses, batch_size])
    mpi_distr = [0]
    final_layer_output_tensor = Tensor[dtype](final_layer_output_traits,
                                              mpi_distr, next_tag)
    next_tag = final_layer_output_tensor.next_tag
    final_layer_output_tensor.from_array(final_layer_output.T)

    final_layer_output_grad = Tensor[dtype](final_layer_output_traits,
                                            mpi_distr, next_tag)
    next_tag = final_layer_output_grad.next_tag
    final_layer_output_tm = TensorMoments(final_layer_output_tensor,
                                          final_layer_output_grad, True)

    # Create crossentropy loss
    xentropy_loss, next_tag = cross_entropy \
        .generate_simple(final_layer_output_tm, next_tag)
    xentropy_loss.y.from_array(true_class_lables)
    xentropy_loss.calc_async()

    nntile_xentropy_np = xentropy_loss.get_val()

    nntile_xentropy_grad_np = xentropy_loss.get_grad()

    xentropy_loss.unregister()
    final_layer_output_tensor.unregister()
    final_layer_output_grad.unregister()
    if dtype == np.float32:
        tol = 1e-5
    elif dtype == np.float64:
        tol = 1e-5

    assert np.max(np.abs(np_xentropy_grad - nntile_xentropy_grad_np.T)) <= tol
    assert np.abs(nntile_xentropy_np[0] - np_xentropy) <= tol
