import pytest

import nntile


@pytest.fixture(scope="package")
def starpu_simple():
    config = nntile.starpu.Config(1, 0, 0)
    nntile.starpu.init()
    return config
