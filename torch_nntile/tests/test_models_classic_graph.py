# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_models_classic_graph.py
# C++ nntile-native models must record classic kernels only (fwd+bwd).
# These are torch_nntile::models (ports of deleted nntile::model::*), not
# Hugging Face torch.nn rewrites.

from __future__ import annotations

import pytest
import torch
from classic_graph import assert_classic_graph

import torch_nntile
from torch_nntile import _C

pytestmark = pytest.mark.skipif(
    not getattr(torch_nntile, "NNTILE_NATIVE_OPS", False),
    reason="classic nntile-native ops not built",
)


def _ids(batch: int = 2, seq: int = 8, vocab: int = 128) -> torch.Tensor:
    return (
        torch.randint(0, vocab, (batch, seq), dtype=torch.long)
        .contiguous()
        .to("nntile")
    )


def _token_types_like(ids: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(ids.cpu()).contiguous().to("nntile")


def _assert_classic_fwd_bwd(
    out: torch.Tensor, *, already_backward: bool = False
) -> None:
    assert out.device.type == "nntile"
    if not already_backward:
        grad = torch.ones(tuple(out.shape), dtype=out.dtype).contiguous()
        out.backward(grad.to(out.device))
    assert_classic_graph()


def test_cpp_llama_classic_graph_fwd_bwd():
    torch_nntile.reset_graph_session()
    out = _C.cpp_llama_causal_forward(
        _ids(),
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        do_backward=True,
    )
    _assert_classic_fwd_bwd(out, already_backward=True)


def test_cpp_gpt2_classic_graph_fwd_bwd():
    torch_nntile.reset_graph_session()
    out = _C.cpp_gpt2_causal_forward(
        _ids(),
        vocab_size=128,
        n_embd=64,
        n_head=4,
        n_layer=1,
        do_backward=True,
    )
    assert out.shape == (2, 8, 128)
    _assert_classic_fwd_bwd(out, already_backward=True)


def test_cpp_gpt_neo_classic_graph_fwd_bwd():
    torch_nntile.reset_graph_session()
    out = _C.cpp_gpt_neo_causal_forward(
        _ids(),
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        window_size=4,
        do_backward=True,
    )
    assert out.shape == (2, 8, 128)
    _assert_classic_fwd_bwd(out, already_backward=True)


def test_cpp_gpt_neox_classic_graph_fwd_bwd():
    torch_nntile.reset_graph_session()
    out = _C.cpp_gpt_neox_causal_forward(
        _ids(),
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        rotary_pct=0.25,
        do_backward=True,
    )
    assert out.shape == (2, 8, 128)
    _assert_classic_fwd_bwd(out, already_backward=True)


def test_cpp_mixer_classic_graph_fwd_bwd():
    torch_nntile.reset_graph_session()
    out = _C.cpp_mlp_mixer_forward(
        torch.randn(8, 2, 4).contiguous().to("nntile"),
        channel_dim=8,
        init_patch_dim=4,
        projected_patch_dim=4,
        num_mixer_layers=1,
        n_classes=3,
        do_backward=True,
    )
    assert out.shape == (2, 3)
    _assert_classic_fwd_bwd(out, already_backward=True)


def test_cpp_deep_relu_classic_graph_fwd_bwd():
    torch_nntile.reset_graph_session()
    x = torch.randn(4, 32).contiguous().to("nntile")
    out = _C.cpp_deep_relu_forward(
        x,
        input_dim=32,
        hidden_dim=64,
        output_dim=8,
        depth=2,
        do_backward=True,
    )
    assert out.shape == (4, 8)
    _assert_classic_fwd_bwd(out, already_backward=True)


def test_cpp_bert_classic_graph_fwd_bwd():
    torch_nntile.reset_graph_session()
    ids = _ids()
    out = _C.cpp_bert_mlm_forward(
        ids,
        _token_types_like(ids),
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        do_backward=True,
    )
    assert out.shape == (2, 8, 128)
    _assert_classic_fwd_bwd(out, already_backward=True)


def test_cpp_roberta_classic_graph_fwd_bwd():
    torch_nntile.reset_graph_session()
    ids_r = torch.randint(4, 128, (2, 8), dtype=torch.long)
    ids_r[0, 0] = 1
    ids_r = ids_r.contiguous().to("nntile")
    out = _C.cpp_roberta_mlm_forward(
        ids_r,
        _token_types_like(ids_r),
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        pad_token_id=1,
        do_backward=True,
    )
    assert out.shape == (2, 8, 128)
    _assert_classic_fwd_bwd(out, already_backward=True)


def test_cpp_t5_classic_graph_fwd_bwd():
    torch_nntile.reset_graph_session()
    out = _C.cpp_t5_forward(
        _ids(),
        _ids(),
        vocab_size=128,
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=1,
        num_heads=4,
        do_backward=True,
    )
    assert out.shape == (2, 8, 128)
    _assert_classic_fwd_bwd(out, already_backward=True)
