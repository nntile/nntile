# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/conftest.py
# Common PyTest configurations for testing
#
# @version 1.1.0

from typing import Generator

import numpy as np
import pytest

import nntile


@pytest.fixture(scope='session')
def context_cpu() -> Generator[None, None, None]:
    context = nntile.Context(ncpu=1, ncuda=1, ooc=0, logger=0, verbose=0)
    context.restrict_cpu()
    try:
        yield None
    finally:
        nntile.starpu.wait_for_all()
        context.shutdown()


@pytest.fixture(scope='session')
def context() -> Generator[None, None, None]:
    context = nntile.Context(ncpu=1, ncuda=1, ooc=0, logger=0, verbose=0)
    context.restrict_cuda()
    try:
        yield None
    finally:
        nntile.starpu.wait_for_all()
        context.shutdown()


@pytest.fixture(scope='function')
def numpy_rng():
    bits = np.random.MT19937(42)
    return np.random.Generator(bits)


@pytest.fixture(scope='function')
def torch_rng():
    import torch
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    gen = torch.Generator()
    gen.manual_seed(42)
    return gen
