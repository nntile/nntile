# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_transformer_layer_parity.py
# Layer / module unit parity: RoPE, RMSNorm, LlamaMLP/Attention, BERT pieces.

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from conftest import nntile_cpu
from parity_helpers import assert_close, clone_to_nntile, contiguous_to_nntile
from torch import Tensor
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

import torch_nntile

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
        sin.unsqueeze(1).expand(batch, heads, seq, head_dim // 2).contiguous()
    )
    cos_exp = (
        cos.unsqueeze(1).expand(batch, heads, seq, head_dim // 2).contiguous()
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
        sin.unsqueeze(1).expand(batch, heads, seq, head_dim // 2).contiguous()
    )
    cos_exp = (
        cos.unsqueeze(1).expand(batch, heads, seq, head_dim // 2).contiguous()
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


def test_rope_heads_as_batch_matches_ref():
    """sin/cos stay ``[B, S, half]``; heads are extra leading modes of x."""
    torch.manual_seed(3)
    heads, batch, seq, head_dim = 4, 2, 8, 16
    position_ids = (
        torch.arange(seq, dtype=torch.long).unsqueeze(0).expand(batch, seq)
    )
    sin, cos = rope_sin_cos_from_position_ids(position_ids, head_dim)
    x = torch.randn(heads, batch, seq, head_dim, dtype=torch.float32)
    grad_y = torch.randn_like(x)
    sin_ref = sin.unsqueeze(0).expand(heads, batch, seq, head_dim // 2)
    cos_ref = cos.unsqueeze(0).expand(heads, batch, seq, head_dim // 2)

    y_ref = _rope_ref_forward(sin_ref, cos_ref, x)
    y_nnt = rope(
        contiguous_to_nntile(sin),
        contiguous_to_nntile(cos),
        contiguous_to_nntile(x),
    )
    assert_close(y_nnt, y_ref, rtol=RTOL, atol=ATOL)

    dx_ref = _rope_ref_backward(sin_ref, cos_ref, grad_y)
    x_nnt = contiguous_to_nntile(x).requires_grad_(True)
    y_nnt = rope(
        contiguous_to_nntile(sin),
        contiguous_to_nntile(cos),
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


def _nntile_linear_cpu(linear: torch.nn.Module, x: Tensor) -> Tensor:
    """CPU ``F.linear`` matching ``NntileLinear`` weight layout."""
    return F.linear(x, linear.weight, linear.bias)


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
        y_cpu = _nntile_linear_cpu(
            mlp.down_proj,
            F.silu(_nntile_linear_cpu(mlp.gate_proj, x))
            * _nntile_linear_cpu(mlp.up_proj, x),
        )

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
    """CPU reference using NNTile Q/K/V/O weight layouts + RoPE + SDPA."""
    b, s, _ = x.shape
    # q_weight: MHA ``[H, D, nh]`` or GQA ``[H, D, n_kv, n_rep]``
    if attn.use_gqa:
        # x @ W -> [B, S, D, n_kv, n_rep] then to [n_kv, n_rep, B, S, D]
        q = torch.einsum("bsh,hdkr->bsdkr", x, attn.q_weight.data)
        q = q.permute(3, 4, 0, 1, 2).contiguous()
        # Flatten kv*rep into head dim for SDPA after rope.
        q_sdpa = q.reshape(attn.n_heads, b, s, attn.head_dim)
    else:
        q = torch.einsum("bsh,hdn->bsdn", x, attn.q_weight.data)
        q_sdpa = q.permute(3, 0, 1, 2).contiguous()

    k = torch.einsum("bsh,hdn->bsdn", x, attn.k_weight.data)
    v = torch.einsum("bsh,hdn->bsdn", x, attn.v_weight.data)
    k_sdpa = k.permute(3, 0, 1, 2).contiguous()
    v_sdpa = v.permute(3, 0, 1, 2).contiguous()

    # RoPE on last dim; broadcast sin/cos over heads.
    def _rope_heads(t: Tensor, n_heads: int) -> Tensor:
        # t: [n_heads, B, S, D]
        sin_h = sin.unsqueeze(0).expand(n_heads, -1, -1, -1).contiguous()
        cos_h = cos.unsqueeze(0).expand(n_heads, -1, -1, -1).contiguous()
        return _rope_ref_forward(sin_h, cos_h, t)

    if attn.use_gqa:
        # RoPE before flattening: apply per (kv, rep) as n_heads.
        q_r = q.reshape(attn.n_heads, b, s, attn.head_dim)
        q_sdpa = _rope_heads(q_r, attn.n_heads)
        k_sdpa = _rope_heads(k_sdpa, attn.n_kv_heads)
        k_sdpa = k_sdpa.repeat_interleave(attn.n_rep, dim=0)
        v_sdpa = v_sdpa.repeat_interleave(attn.n_rep, dim=0)
    else:
        q_sdpa = _rope_heads(q_sdpa, attn.n_heads)
        k_sdpa = _rope_heads(k_sdpa, attn.n_kv_heads)

    # F.sdpa expects [B, nh, S, D]
    q_b = q_sdpa.permute(1, 0, 2, 3)
    k_b = k_sdpa.permute(1, 0, 2, 3)
    v_b = v_sdpa.permute(1, 0, 2, 3)
    out = F.scaled_dot_product_attention(
        q_b, k_b, v_b, attn_mask=None, dropout_p=0.0, is_causal=True
    )
    # back to [nh, B, S, D] then O proj
    out = out.permute(1, 0, 2, 3).contiguous()
    if attn.use_gqa:
        # o_weight: [D, n_kv, n_rep, H]
        out = out.reshape(attn.n_kv_heads, attn.n_rep, b, s, attn.head_dim)
        y = torch.einsum("krbsd,dkrh->bsh", out, attn.o_weight.data)
    else:
        # o_weight: [D, nh, H]; out [nh, B, S, D]
        y = torch.einsum("nbsd,dnh->bsh", out, attn.o_weight.data)
    return y


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


def _bert_self_attn_cpu_ref(attn: BertSelfAttention, x: Tensor) -> Tensor:
    """CPU reference for NNTile QKV projection + dense SDPA (no mask)."""
    b, s, _ = x.shape
    nh, hs = attn.n_heads, attn.head_dim

    def _proj(layer) -> Tensor:
        # weight [H, hs, nh] -> [nh, B, S, hs]
        y = torch.einsum("bsh,hdn->bsdn", x, layer.weight.data)
        y = y.permute(3, 0, 1, 2).contiguous()
        if layer.bias is not None:
            y = y + layer.bias.data.view(nh, 1, 1, hs)
        return y

    q = _proj(attn.query)
    k = _proj(attn.key)
    v = _proj(attn.value)
    q_b = q.permute(1, 0, 2, 3)
    k_b = k.permute(1, 0, 2, 3)
    v_b = v.permute(1, 0, 2, 3)
    out = F.scaled_dot_product_attention(
        q_b, k_b, v_b, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    # BertSelfAttention returns SDPA kernel layout [nh, B, S, hs]
    return out.permute(1, 0, 2, 3).contiguous()


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
        y_cpu = _bert_self_attn_cpu_ref(attn, x)

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
    with torch.no_grad():
        y_shape = _bert_self_attn_cpu_ref(attn, x).shape
    grad_out = torch.randn(y_shape, dtype=torch.float32)

    attn_cpu = BertSelfAttention(cfg).float()
    attn_cpu.load_state_dict(attn.state_dict())
    for p in attn_cpu.parameters():
        p.requires_grad_(True)
    x_cpu = x.detach().requires_grad_(True)

    def _proj(layer, inp):
        nh, hs = attn_cpu.n_heads, attn_cpu.head_dim
        y = torch.einsum("bsh,hdn->bsdn", inp, layer.weight)
        y = y.permute(3, 0, 1, 2).contiguous()
        if layer.bias is not None:
            y = y + layer.bias.view(nh, 1, 1, hs)
        return y

    q = _proj(attn_cpu.query, x_cpu)
    k = _proj(attn_cpu.key, x_cpu)
    v = _proj(attn_cpu.value, x_cpu)
    out = F.scaled_dot_product_attention(
        q.permute(1, 0, 2, 3),
        k.permute(1, 0, 2, 3),
        v.permute(1, 0, 2, 3),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    ).permute(1, 0, 2, 3)
    out.backward(grad_out)

    attn_nnt = clone_to_nntile(attn)
    for p in attn_nnt.parameters():
        p.requires_grad_(True)
    x_nnt = contiguous_to_nntile(x).requires_grad_(True)
    y_nnt = attn_nnt(x_nnt, is_causal=False)
    y_nnt.backward(contiguous_to_nntile(grad_out))

    assert_close(x_nnt.grad, x_cpu.grad, rtol=RTOL, atol=ATOL)
    assert_close(
        attn_nnt.query.weight.grad,
        attn_cpu.query.weight.grad,
        rtol=RTOL,
        atol=ATOL,
    )
