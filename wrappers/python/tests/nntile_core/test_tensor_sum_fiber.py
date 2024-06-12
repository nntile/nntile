# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/nntile_core/test_tensor_sum_fiber.py
# Test for tensor::sum_fiber<T> Python wrapper
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-04-26

import pytest


@pytest.mark.xfail(reason='not implemented')
def test_sum_fiber_async():
    pass
