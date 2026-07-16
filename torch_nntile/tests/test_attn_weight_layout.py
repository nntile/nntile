# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_attn_weight_layout.py
# Attention weight layout conversion tests.

from __future__ import annotations

import torch
from torch_nntile.nn.weight_layout import (
    convert_attn_weights,
    nntile_to_torch_o_weight,
    nntile_to_torch_qkv_weight,
    torch_to_nntile_o_weight,
    torch_to_nntile_qkv_weight,
)


def test_qkv_weight_round_trip():
    hidden, n_heads, head_size = 32, 4, 8
    w = torch.randn(hidden, n_heads, head_size)
    w_nt = torch_to_nntile_qkv_weight(w)
    assert w_nt.shape == (hidden, head_size, n_heads)
    w_rt = nntile_to_torch_qkv_weight(w_nt)
    assert torch.equal(w, w_rt)


def test_o_weight_round_trip():
    n_heads, head_size, hidden = 4, 8, 32
    w = torch.randn(n_heads, head_size, hidden)
    w_nt = torch_to_nntile_o_weight(w)
    assert w_nt.shape == (head_size, n_heads, hidden)
    w_rt = nntile_to_torch_o_weight(w_nt)
    assert torch.equal(w, w_rt)


def test_convert_attn_weights_torch_to_nntile():
    weights = {
        "attn.q_weight": torch.randn(32, 4, 8),
        "attn.k_weight": torch.randn(32, 4, 8),
        "attn.v_weight": torch.randn(32, 4, 8),
        "attn.o_weight": torch.randn(4, 8, 32),
        "attn.q_bias": torch.randn(4, 8),
    }
    out = convert_attn_weights(weights, "torch_to_nntile")
    assert out["attn.q_weight"].shape == (32, 8, 4)
    assert out["attn.o_weight"].shape == (8, 4, 32)
    assert torch.equal(out["attn.q_bias"], weights["attn.q_bias"])


def test_convert_attn_weights_matches_generate_test_data_layout():
    """Match ``generate_test_data._gpt2_attn_weights`` axis swaps."""
    hidden, n_heads, head_size = 64, 4, 16
    # HF-style stacked c_attn slice as (hidden, n_heads, head_size)
    hf_q = torch.randn(hidden, n_heads, head_size)
    nnt_q = hf_q.transpose(1, 2).contiguous()
    assert torch.equal(torch_to_nntile_qkv_weight(hf_q), nnt_q)

    hf_o = torch.randn(n_heads, head_size, hidden)
    nnt_o = hf_o.transpose(0, 1).contiguous()
    assert torch.equal(torch_to_nntile_o_weight(hf_o), nnt_o)

    round_trip = convert_attn_weights(
        {
            "attn.q_weight": hf_q,
            "attn.o_weight": hf_o,
        },
        "torch_to_nntile",
    )
    back = convert_attn_weights(round_trip, "nntile_to_torch")
    assert torch.equal(back["attn.q_weight"], hf_q)
    assert torch.equal(back["attn.o_weight"], hf_o)
