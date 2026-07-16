# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_cpp_models_smoke.py
# Smoke-test C++ libtorch_nntile models via Python ``_C`` bindings.

from __future__ import annotations

import pytest
import torch

from torch_nntile import _C


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)


def test_cpp_models_listed():
    names = set(_C.cpp_models_listed())
    assert "LlamaCausal" in names
    assert "BertMlm" in names
    assert "RobertaMlm" in names
    assert "GptNeoCausal" in names
    assert "GptNeoXCausal" in names
    assert "Gpt2Causal" in names
    assert "T5" in names


def test_cpp_llama_causal_forward_on_nntile():
    ids = torch.randint(0, 128, (2, 8), dtype=torch.long).contiguous().to(
        "nntile"
    )
    out = _C.cpp_llama_causal_forward(
        ids,
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    assert out.device.type == "nntile"
    assert tuple(out.shape) == (2, 8, 128)


def test_cpp_bert_mlm_forward_on_nntile():
    ids = torch.randint(0, 128, (2, 8), dtype=torch.long).contiguous().to(
        "nntile"
    )
    # zeros_like on nntile long fails (fill_ is float32-only in graph mode).
    tt = torch.zeros_like(ids.cpu()).contiguous().to("nntile")
    out = _C.cpp_bert_mlm_forward(
        ids,
        tt,
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
    )
    assert out.device.type == "nntile"
    assert tuple(out.shape) == (2, 8, 128)


def test_cpp_roberta_mlm_forward_on_nntile():
    ids = torch.randint(4, 128, (2, 8), dtype=torch.long)
    ids[0, 0] = 1  # pad
    ids = ids.contiguous().to("nntile")
    tt = torch.zeros_like(ids.cpu()).contiguous().to("nntile")
    out = _C.cpp_roberta_mlm_forward(
        ids,
        tt,
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        pad_token_id=1,
    )
    assert out.device.type == "nntile"
    assert tuple(out.shape) == (2, 8, 128)


def test_cpp_gpt_neo_causal_forward_on_nntile():
    ids = torch.randint(0, 128, (2, 8), dtype=torch.long).contiguous().to(
        "nntile"
    )
    out = _C.cpp_gpt_neo_causal_forward(
        ids,
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        window_size=4,
    )
    assert out.device.type == "nntile"
    assert tuple(out.shape) == (2, 8, 128)


def test_cpp_gpt_neox_causal_forward_on_nntile():
    ids = torch.randint(0, 128, (2, 8), dtype=torch.long).contiguous().to(
        "nntile"
    )
    out = _C.cpp_gpt_neox_causal_forward(
        ids,
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        rotary_pct=0.25,
    )
    assert out.device.type == "nntile"
    assert tuple(out.shape) == (2, 8, 128)


def test_cpp_gpt2_causal_forward_on_nntile():
    ids = torch.randint(0, 128, (2, 8), dtype=torch.long).contiguous().to(
        "nntile"
    )
    out = _C.cpp_gpt2_causal_forward(
        ids,
        vocab_size=128,
        n_embd=64,
        n_head=4,
        n_layer=1,
    )
    assert out.device.type == "nntile"
    assert tuple(out.shape) == (2, 8, 128)


def test_cpp_t5_forward_on_nntile():
    # Equal enc/dec seq: nntile SDPA currently requires Q/K/V same shape
    # (cross-attn with unequal lengths is not supported yet).
    enc = torch.randint(0, 128, (2, 8), dtype=torch.long).contiguous().to(
        "nntile"
    )
    dec = torch.randint(0, 128, (2, 8), dtype=torch.long).contiguous().to(
        "nntile"
    )
    out = _C.cpp_t5_forward(
        enc,
        dec,
        vocab_size=128,
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=1,
        num_heads=4,
    )
    assert out.device.type == "nntile"
    assert tuple(out.shape) == (2, 8, 128)
