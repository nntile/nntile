# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_gelu_backward.py
# Test for tensor::gelu_backward<T> Python wrapper
#
# @version 1.1.0

import pytest


@pytest.mark.xfail(reason='not implemented')
def test_gelu_backward_async(context):
    pass
