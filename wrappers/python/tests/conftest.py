import os
from typing import Generator

import pytest

import nntile


@pytest.fixture(scope="session")
def starpu_simple() -> Generator[nntile.starpu.Config, None, None]:
    config = nntile.starpu.Config(1, 1, 1)
    nntile.starpu.init()
    nntile.starpu.restrict_cpu()
    try:
        yield config
    finally:
        nntile.starpu.wait_for_all()


@pytest.fixture(scope="package")
def numpy_rng():
    import random

    import numpy as np

    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    bits = np.random.MT19937()
    rng = np.random.Generator(bits)
    return rng


@pytest.fixture(scope="package")
def torch_rng():
    import torch

    seed = 42

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    gen = torch.Generator()
    gen.manual_seed(seed)

    return gen


@pytest.fixture(scope="package")
def huggingface_local_cache_dir():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return f"{dir_path}/hf_cache"
