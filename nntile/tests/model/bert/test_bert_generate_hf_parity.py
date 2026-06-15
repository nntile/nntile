#!/usr/bin/env python3
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file nntile/tests/model/bert/test_bert_generate_hf_parity.py
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
    _make_config, _out_to_nntile, as_float32, generate_attention,
    generate_intermediate)


def _hf_attn_qkv_weight(
    linear: torch.nn.Linear, n_emb: int, nh: int, hs: int,
) -> np.ndarray:
    """Explicit HF PT Linear → graph Q/K/V layout (graph ``(H, hd, nh)``)."""
    w = linear.weight.detach().numpy().reshape(nh, hs, n_emb)
    return as_float32(w.transpose(2, 1, 0))


def _hf_attn_o_weight(
    linear: torch.nn.Linear, n_emb: int, nh: int, hs: int,
) -> np.ndarray:
    """Explicit HF PT Linear → graph output-dense layout (graph ``(hd, nh, H)``)."""
    w = linear.weight.detach().numpy().reshape(n_emb, nh, hs)
    return as_float32(w.transpose(2, 1, 0))


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
    H = ATTENTION_DIMS.hidden
    nh = ATTENTION_DIMS.n_heads
    hs = ATTENTION_DIMS.head_size

    for key, linear in (
        ("q_weight", pt.self.query),
        ("k_weight", pt.self.key),
        ("v_weight", pt.self.value),
    ):
        expected = _hf_attn_qkv_weight(linear, H, nh, hs)
        assert np.array_equal(weights[f"attn.self.{key}"], expected)

    expected_o = _hf_attn_o_weight(pt.output.dense, H, nh, hs)
    assert np.array_equal(
        weights["attn.output.dense.weight"], expected_o,
    )


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
