# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file python/tests/test_import.py
#
# @version 1.1.0

"""Smoke import tests for the nntile extension."""

from __future__ import annotations

import nntile


def test_public_api_imports():
    assert hasattr(nntile, 'Context')
    assert hasattr(nntile, 'NNGraph')
    assert hasattr(nntile, 'Runtime')
    assert hasattr(nntile, 'Mlp')
    assert hasattr(nntile.nn, 'gemm')


def test_all_exports():
    expected = {
        'Context',
        'NNGraph',
        'Runtime',
        'TensorGraph',
        'TileGraph',
        'Mlp',
    }
    missing = expected - set(nntile.__all__)
    assert not missing, f'missing from __all__: {missing}'
