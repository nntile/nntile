# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_sumprod_slice.py
# Test for tensor::sumprod_slice<T> Python wrapper
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-04-26

import pytest


@pytest.mark.xfail(reason='not implemented')
def test_sumprod_slice_async():
    pass
