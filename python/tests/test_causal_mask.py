# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file python/tests/test_causal_mask.py
#
# @version 1.1.0

"""Causal mask layout matches nntile/nn/ops/sdpa_causal_mask.cc."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.numpy_only

_examples = Path(__file__).resolve().parents[1] / 'examples'
if str(_examples) not in sys.path:
    sys.path.insert(0, str(_examples))

from numpy_helpers import sdpa_causal_mask_bool_fill


def reference_mask(seq_len: int) -> np.ndarray:
    allowed = np.zeros((seq_len, seq_len), dtype=np.uint8)
    for key in range(seq_len):
        for query in range(seq_len):
            if key <= query:
                allowed[key, query] = 1
    return np.ascontiguousarray(allowed.T).ravel()


def test_sdpa_causal_mask_matches_cpp_layout():
    for n_seq in (1, 4, 8):
        got = sdpa_causal_mask_bool_fill(n_seq)
        ref = reference_mask(n_seq)
        assert got.shape == (n_seq * n_seq,)
        np.testing.assert_array_equal(got, ref)
        for query in range(n_seq):
            for key in range(n_seq):
                val = got[query * n_seq + key]
                assert val == (1 if key <= query else 0)
