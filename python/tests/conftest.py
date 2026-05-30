# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file python/tests/conftest.py
#
# @version 1.1.0

"""Pytest fixtures for the NNTile Python package."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope='session')
def nntile_context():
    """Initialize StarPU once per test session."""
    nntile = pytest.importorskip('nntile')
    ctx = nntile.Context(1, 0)
    yield ctx
    ctx.shutdown()


@pytest.fixture(autouse=True)
def _check_libs(request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker('numpy_only') is not None:
        return
    if 'LD_LIBRARY_PATH' not in os.environ:
        pytest.skip(
            'Set LD_LIBRARY_PATH to libnntile.so and StarPU before running tests',
        )
