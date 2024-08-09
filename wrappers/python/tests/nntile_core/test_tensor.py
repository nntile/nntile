# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor.py
# Test for Tensor<T> Python wrapper
#
# @version 1.1.0

import numpy as np
import pytest
from numpy.testing import assert_equal

import nntile

config = nntile.starpu.Config(1, 0, 0)
nntile.starpu.init()

# Define mapping between numpy and nntile types
Tensor = {np.float32: nntile.tensor.Tensor_fp32,
          np.float64: nntile.tensor.Tensor_fp64}


class TestTensor:

    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    def test_init(self, dtype):
        shape = [3, 4]
        mpi_distr = [0]
        next_tag = 0
        traits = nntile.tensor.TensorTraits(shape, shape)
        tensor = Tensor[dtype](traits, mpi_distr, next_tag)
        src = np.random.default_rng(42) \
            .standard_normal(shape) \
            .astype(dtype, 'F')
        dst = np.zeros_like(src)
        tensor.from_array(src)
        tensor.to_array(dst)
        nntile.starpu.wait_for_all()
        tensor.unregister()
        assert_equal(dst, src)
