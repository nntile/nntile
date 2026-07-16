# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_transformer_layer_parity.py
# Layer / module unit parity: RoPE, RMSNorm, LlamaMLP/Attention, BERT pieces.

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

import torch_nntile
from torch_nntile import _C
from torch_nntile.models.bert import BertConfig, BertSelfAttention
from torch_nntile.models.llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaMLP,
    LlamaRMSNorm,
)
from torch_nntile.normalization import rms_norm
from torch_nntile.rope import (
    _rope_ref_backward,
    _rope_ref_forward,
    rope,
    rope_sin_cos_from_position_ids,
)
from conftest import nntile_cpu
from parity_helpers import assert_close, clone_to_nntile, contiguous_to_nntile


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)

RTOL = 1e-4
ATOL = 1e-4


def _cpu_rms_norm(
    x: Tensor,
    weight: Tensor,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Simple CPU RMSNorm matching ``F.rms_norm`` / LlamaRMSNorm math."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return weight * x_normed


# ---------------------------------------------------------------------------
# A. RoPE
# ---------------------------------------------------------------------------


def test_rope_forward_matches_ref():
    torch.manual_seed(0)
    batch, heads, seq, head_dim = 2, 4, 8, 16
    position_ids = (
        torch.arange(seq, dtype=torch.long).unsqueeze(0).expand(batch, seq)
    )
    sin, cos = rope_sin_cos_from_position_ids(position_ids, head_dim)
    x = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)

    sin_exp = (
        sin.unsqueeze(1)
        .expand(batch, heads, seq, head_dim // 2)
        .contiguous()
    )
    cos_exp = (
        cos.unsqueeze(1)
        .expand(batch, heads, seq, head_dim // 2)
        .contiguous()
    )

    y_ref = _rope_ref_forward(sin_exp, cos_exp, x)
    y_nnt = rope(
        contiguous_to_nntile(sin_exp),
        contiguous_to_nntile(cos_exp),
        contiguous_to_nntile(x),
    )
    assert_close(y_nnt, y_ref, rtol=RTOL, atol=ATOL)


def test_rope_backward_matches_ref():
    torch.manual_seed(1)
    batch, heads, seq, head_dim = 2, 4, 8, 16
    position_ids = (
        torch.arange(seq, dtype=torch.long).unsqueeze(0).expand(batch, seq)
    )
    sin, cos = rope_sin_cos_from_position_ids(position_ids, head_dim)
    x = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    grad_y = torch.randn_like(x)

    sin_exp = (
        sin.unsqueeze(1)
        .expand(batch, heads, seq, head_dim // 2)
        .contiguous()
    )
    cos_exp = (
        cos.unsqueeze(1)
        .expand(batch, heads, seq, head_dim // 2)
        .contiguous()
    )

    dx_ref = _rope_ref_backward(sin_exp, cos_exp, grad_y)

    x_nnt = contiguous_to_nntile(x).requires_grad_(True)
    y_nnt = rope(
        contiguous_to_nntile(sin_exp),
        contiguous_to_nntile(cos_exp),
        x_nnt,
    )
    (dx_nnt,) = torch.autograd.grad(
        y_nnt,
        x_nnt,
        grad_outputs=contiguous_to_nntile(grad_y),
    )
    assert_close(dx_nnt, dx_ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# B. RMSNorm
# ---------------------------------------------------------------------------


def test_rms_norm_functional_matches_cpu():
    torch.manual_seed(2)
    x = torch.randn(2, 8, 64, dtype=torch.float32)
    weight = torch.randn(64, dtype=torch.float32) * 0.1 + 1.0
    eps = 1e-6

    y_ref = _cpu_rms_norm(x, weight, eps=eps)
    y_nnt = rms_norm(
        contiguous_to_nntile(x),
        (64,),
        contiguous_to_nntile(weight),
        eps,
    )
    assert_close(y_nnt, y_ref, rtol=RTOL, atol=ATOL)


def test_llama_rms_norm_matches_cpu():
    torch.manual_seed(3)
    hidden = 64
    x = torch.randn(2, 8, hidden, dtype=torch.float32)
    mod = LlamaRMSNorm(hidden, eps=1e-6).float()
    with torch.no_grad():
        mod.weight.normal_(mean=1.0, std=0.05)

    y_ref = _cpu_rms_norm(x, mod.weight.data, eps=mod.eps)
    mod_nnt = clone_to_nntile(mod)
    y_nnt = mod_nnt(contiguous_to_nntile(x))
    assert_close(y_nnt, y_ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# C. LlamaMLP
# ---------------------------------------------------------------------------


def test_llama_mlp_forward_matches_cpu():
    torch.manual_seed(4)
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        mlp_bias=False,
    )
    mlp = LlamaMLP(cfg).float().eval()
    x = torch.randn(2, 8, cfg.hidden_size, dtype=torch.float32)

    with torch.no_grad():
        y_cpu = mlp(x)

    mlp_nnt = clone_to_nntile(mlp)
    with torch.no_grad():
        y_nnt = mlp_nnt(contiguous_to_nntile(x))
    assert_close(y_nnt, y_cpu, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# D. LlamaAttention (MHA + GQA)
# ---------------------------------------------------------------------------


def _llama_attn_cpu_ref(
    attn: LlamaAttention,
    x: Tensor,
    sin: Tensor,
    cos: Tensor,
) -> Tensor:
    """CPU reference: same Linear weights + RoPE + SDPA (+ GQA repeat)."""
    b, s, _ = x.shape
    q = attn._shape(attn.q_proj(x), attn.n_heads)
    k = attn._shape(attn.k_proj(x), attn.n_kv_heads)
    v = attn._shape(attn.v_proj(x), attn.n_kv_heads)

    n_heads = q.size(1)
    sin_h = sin.unsqueeze(1).expand(-1, n_heads, -1, -1).contiguous()
    cos_h = cos.unsqueeze(1).expand(-1, n_heads, -1, -1).contiguous()
    # KV heads may differ under GQA — expand sin/cos to kv head count too.
    sin_k = sin.unsqueeze(1).expand(-1, attn.n_kv_heads, -1, -1).contiguous()
    cos_k = cos.unsqueeze(1).expand(-1, attn.n_kv_heads, -1, -1).contiguous()
    q = _rope_ref_forward(sin_h, cos_h, q)
    k = _rope_ref_forward(sin_k, cos_k, k)

    if attn.n_rep > 1:
        k = k.repeat_interleave(attn.n_rep, dim=1)
        v = v.repeat_interleave(attn.n_rep, dim=1)

    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
    )
    out = out.transpose(1, 2).contiguous().view(b, s, -1)
    return attn.o_proj(out)


@pytest.mark.parametrize(
    "n_heads,n_kv,head_dim",
    [
        (2, 2, 8),  # MHA
        (2, 1, 8),  # GQA
    ],
)
def test_llama_attention_forward_matches_cpu(n_heads, n_kv, head_dim):
    torch.manual_seed(5 + n_kv)
    hidden = n_heads * head_dim
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=1,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        max_position_embeddings=32,
        attention_bias=False,
    )
    attn = LlamaAttention(cfg).float().eval()
    batch, seq = 2, 8
    x = torch.randn(batch, seq, hidden, dtype=torch.float32)
    position_ids = (
        torch.arange(seq, dtype=torch.long).unsqueeze(0).expand(batch, seq)
    )
    sin, cos = rope_sin_cos_from_position_ids(position_ids, head_dim)

    with torch.no_grad():
        y_ref = _llama_attn_cpu_ref(attn, x, sin, cos)

    attn_nnt = clone_to_nntile(attn)
    with torch.no_grad():
        y_nnt = attn_nnt(
            contiguous_to_nntile(x),
            sin=contiguous_to_nntile(sin),
            cos=contiguous_to_nntile(cos),
            is_causal=True,
        )
    assert_close(y_nnt, y_ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# E. BertSelfAttention
# ---------------------------------------------------------------------------


def test_bert_self_attention_forward_matches_cpu():
    torch.manual_seed(7)
    cfg = BertConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=32,
    )
    attn = BertSelfAttention(cfg).float().eval()
    x = torch.randn(2, 8, cfg.hidden_size, dtype=torch.float32)

    with torch.no_grad():
        y_cpu = attn(x, is_causal=False)

    attn_nnt = clone_to_nntile(attn)
    with torch.no_grad():
        y_nnt = attn_nnt(contiguous_to_nntile(x), is_causal=False)
    assert_close(y_nnt, y_cpu, rtol=RTOL, atol=ATOL)


def test_bert_self_attention_backward_matches_cpu():
    torch.manual_seed(8)
    cfg = BertConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=32,
    )
    attn = BertSelfAttention(cfg).float()
    x = torch.randn(2, 8, cfg.hidden_size, dtype=torch.float32)
    grad_out = torch.randn_like(x)

    x_cpu = x.detach().requires_grad_(True)
    y_cpu = attn(x_cpu, is_causal=False)
    y_cpu.backward(grad_out)

    attn_nnt = clone_to_nntile(attn)
    for p in attn_nnt.parameters():
        p.requires_grad_(True)
    x_nnt = contiguous_to_nntile(x).requires_grad_(True)
    y_nnt = attn_nnt(x_nnt, is_causal=False)
    y_nnt.backward(contiguous_to_nntile(grad_out))

    assert_close(x_nnt.grad, x_cpu.grad, rtol=RTOL, atol=ATOL)
    assert_close(
        attn_nnt.query.weight.grad,
        attn.query.weight.grad,
        rtol=RTOL,
        atol=ATOL,
    )
