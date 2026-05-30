# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file python/examples/numpy_helpers.py
# Pure-NumPy helpers shared by graph training examples and tests.
#
# @version 1.1.0

"""NumPy-only utilities for causal LM example scripts (no nntile import)."""

from __future__ import annotations

import numpy as np


def fill_arange_position_ids(
    pos_data: np.ndarray, n_seq: int, n_batch: int,
) -> None:
    for b in range(n_batch):
        for s in range(n_seq):
            pos_data[s + n_seq * b] = s


def sdpa_causal_mask_bool_fortran_fill(n_seq: int) -> np.ndarray:
    """BOOL causal mask, Fortran layout: out[kk + n_seq * qq] = (kk <= qq)."""
    out = np.zeros(n_seq * n_seq, dtype=np.uint8)
    for qq in range(n_seq):
        for kk in range(n_seq):
            if kk <= qq:
                out[kk + n_seq * qq] = 1
    return out
