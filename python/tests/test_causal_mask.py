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

_examples = Path(__file__).resolve().parents[1] / 'examples'
if str(_examples) not in sys.path:
    sys.path.insert(0, str(_examples))

from numpy_helpers import sdpa_causal_mask_bool_fortran_fill


def reference_mask_fortran(seq_len: int) -> np.ndarray:
    out = np.zeros(seq_len * seq_len, dtype=np.uint8)
    for qq in range(seq_len):
        for kk in range(seq_len):
            if kk <= qq:
                out[kk + seq_len * qq] = 1
    return out


def test_sdpa_causal_mask_matches_cpp_layout():
    for n_seq in (1, 4, 8):
        got = sdpa_causal_mask_bool_fortran_fill(n_seq)
        ref = reference_mask_fortran(n_seq)
        assert got.shape == (n_seq * n_seq,)
        np.testing.assert_array_equal(got, ref)
        # Lower triangle in (key, query) Fortran indexing.
        for qq in range(n_seq):
            for kk in range(n_seq):
                val = got[kk + n_seq * qq]
                assert val == (1 if kk <= qq else 0)
