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
import torch

import nntile
from nntile.loss import CrossEntropy as cross_entropy
from nntile.tensor import TensorMoments

nntile.nntile_init(
    ncpus=1,
    ncuda=0,
    cublas=0,
    ooc=0,
    logger=0,
    verbose=0,
)

dtype2nntile = {
        'fp32': nntile.tensor.Tensor_fp32,
        'fp32_fast_tf32': nntile.tensor.Tensor_fp32_fast_tf32,
        'bf16': nntile.tensor.Tensor_bf16,
        'fp32_fast_bf16': nntile.tensor.Tensor_fp32_fast_bf16,
        'fp32_fast_fp16': nntile.tensor.Tensor_fp32_fast_fp16,
}

dtype2tol = {
        'fp32': {'rtol': 1e-6},
        'fp32_fast_tf32': {'rtol': 8e-4},
        'bf16': {'rtol': 1.6e-2},
        'fp32_fast_bf16': {'rtol': 8e-4},
        'fp32_fast_fp16': {'rtol': 8e-4},
}

nocuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no cuda')


@pytest.mark.parametrize('dtype', [
    'fp32',
    pytest.param('fp32_fast_tf32', marks=nocuda),
    pytest.param('bf16', marks=nocuda),
    pytest.param('fp32_fast_bf16', marks=nocuda),
    pytest.param('fp32_fast_fp16', marks=nocuda),
    ])
@pytest.mark.parametrize('reduction', [
                          "sum",
                          "mean"
                          ])
@pytest.mark.parametrize('ignore_index', [
    True,
    False
])
def test_cross_entropy(dtype: np.dtype, reduction: str,
                       ignore_index: bool):
    """Helper function returns bool value true if test passes."""
    rng = np.random.default_rng(42)

    # Describe single-tile tensor, located at node 0
    nclasses = 5
    batch_size = 7
    final_layer_output = rng.standard_normal((batch_size, nclasses),
                                             dtype=np.float32)
    final_layer_output = np.array(final_layer_output, order="F")
    final_layer_output_torch = torch.tensor(final_layer_output,
                                            requires_grad=True)

    true_class_lables = rng.integers(0, nclasses,
                                     (batch_size,), dtype=np.int64)
    true_class_lables = np.array(true_class_lables, order="F")
    if ignore_index:
        true_class_lables[rng.uniform(size=(batch_size,)) < 0.5] = -100
    true_class_lables_torch = torch.tensor(true_class_lables, dtype=torch.long)

    torch_xentropy_loss = torch.nn.CrossEntropyLoss(reduction=reduction)

    xentropy_val_torch = torch_xentropy_loss(final_layer_output_torch,
                                             true_class_lables_torch)
    xentropy_val_torch.backward()

    x_type = dtype2nntile[dtype]
    final_layer_output_traits = nntile.tensor.TensorTraits(
        [nclasses, batch_size], [nclasses, batch_size])
    final_layer_output_tensor = x_type(final_layer_output_traits)
    final_layer_output_tensor.from_array(final_layer_output.T)

    final_layer_grad_output_traits = nntile.tensor.TensorTraits(
        [nclasses, batch_size], [nclasses, batch_size])
    final_layer_output_grad = x_type(final_layer_grad_output_traits)
    final_layer_output_tm = TensorMoments(final_layer_output_tensor,
                                          final_layer_output_grad, True)

    # Create crossentropy loss
    if reduction == "sum":
        xentropy_loss = cross_entropy.generate_simple(final_layer_output_tm)
    elif reduction == "mean":
        xentropy_loss = cross_entropy.generate_simple(
                                final_layer_output_tm,
                                scale=1. / np.sum(true_class_lables != -100))
    xentropy_loss.y.from_array(true_class_lables)
    xentropy_loss.calc_async()

    nntile_xentropy_nnntile = xentropy_loss.get_val()

    nntile_xentropy_grad_np = xentropy_loss.get_grad()

    xentropy_loss.unregister()
    final_layer_output_tm.unregister()
    rtol = dtype2tol[dtype]["rtol"]
    assert np.max(np.abs(final_layer_output_torch.grad.numpy() -
                         nntile_xentropy_grad_np.T)) <= \
        rtol * np.max(np.abs(final_layer_output_torch.grad.numpy()))
    assert np.abs(nntile_xentropy_nnntile[0] - xentropy_val_torch.item()) <= \
        rtol * xentropy_val_torch.item()
