#!/usr/bin/env python3
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file tests/graph/model/bert/test_bert_generate_hf_parity.py
# Regression: BERT generate_test_data uses full HF forwards.
#
# @version 1.1.0

"""Guard BERT graph fixtures track HuggingFace ``modeling_bert``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from transformers.models.bert.modeling_bert import (
    BertAttention, BertIntermediate)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_test_data import (  # noqa: E402
    ATTENTION_DIMS, INTERMEDIATE_DIMS, _bert_attention_weights, _hidden_input,
    _make_config, _out_to_nntile, fortran_order, generate_attention,
    generate_intermediate)


def test_attention_fixture_matches_hf_forward() -> None:
    seed = 42
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(ATTENTION_DIMS)
    pt = BertAttention(config).eval()
    _, x_pt = _hidden_input(rng, ATTENTION_DIMS)

    hf_ref = _out_to_nntile(pt(x_pt)[0])
    bundle = generate_attention(seed)
    assert np.allclose(bundle["output_ref"], hf_ref, rtol=1e-6, atol=1e-6)


def test_attention_weights_match_hf_layout() -> None:
    config = _make_config(ATTENTION_DIMS)
    pt = BertAttention(config).eval()
    weights = _bert_attention_weights(pt, "attn", ATTENTION_DIMS)
    w_q = (
        pt.self.query.weight.detach()
        .numpy()
        .reshape(
            ATTENTION_DIMS.n_heads,
            ATTENTION_DIMS.head_size,
            ATTENTION_DIMS.hidden,
        )
    )
    assert np.array_equal(weights["attn.self.q_weight"], fortran_order(w_q))


def test_intermediate_matches_hf() -> None:
    seed = 7
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = _make_config(INTERMEDIATE_DIMS)
    pt = BertIntermediate(config).eval()
    _, x_pt = _hidden_input(rng, INTERMEDIATE_DIMS)
    hf_ref = _out_to_nntile(pt(x_pt))
    bundle = generate_intermediate(seed)
    assert np.allclose(bundle["output_ref"], hf_ref, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    test_attention_fixture_matches_hf_forward()
    test_attention_weights_match_hf_layout()
    test_intermediate_matches_hf()
    print("ok")
