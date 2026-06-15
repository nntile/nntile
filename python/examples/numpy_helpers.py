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
            pos_data[b * n_seq + s] = s


def sdpa_causal_mask_bool_fill(n_seq: int) -> np.ndarray:
    """BOOL causal mask for ``[seq, seq]`` NNGraph bind (graph).

    Logical ``mask[key, query] = (key <= query)``; stored as ``mask.T`` for
    bind layout compatible with legacy shape labels.
    """
    out = np.zeros(n_seq * n_seq, dtype=np.uint8)
    for qq in range(n_seq):
        for kk in range(n_seq):
            if kk <= qq:
                out[qq * n_seq + kk] = 1
    return out
