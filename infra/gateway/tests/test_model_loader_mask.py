"""Regression: BERT padding mask flat layout matches sdpa_eager bind."""

from __future__ import annotations

import numpy as np

from nntile_gateway.model_loader import _build_padding_mask


def test_padding_mask_flat_layout_matches_sdpa_convention() -> None:
    seq_len = 6
    actual_len = 3
    mask = _build_padding_mask(seq_len, actual_len)

    keep = np.zeros(seq_len, dtype=bool)
    keep[:actual_len] = True

    # Logical mask[key, query] = keep[key]; sdpa uses out[query * seq + key].
    for query in range(seq_len):
        for key in range(seq_len):
            flat = query * seq_len + key
            assert bool(mask.ravel()[flat]) == bool(keep[key])
