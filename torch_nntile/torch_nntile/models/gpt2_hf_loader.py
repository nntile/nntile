# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/gpt2_hf_loader.py
# Load HuggingFace GPT-2 weights into minimal torch_nntile GPT2LMHead.

"""Convert HF ``GPT2LMHeadModel`` weights into NNTile-layout minimal GPT-2."""

from __future__ import annotations

import torch
from transformers import GPT2LMHeadModel

from torch_nntile.nn.weight_layout import (
    torch_to_nntile_o_weight,
    torch_to_nntile_qkv_weight,
)
from torch_nntile.models.gpt2_minimal import GPT2LMHead


def _conv1d_to_linear_weight(weight: torch.Tensor) -> torch.Tensor:
    """HF Conv1D ``(in, out)`` -> addmm ``(out, in)``."""
    return weight.t().contiguous()


def _split_hf_attn_weights(
    attn,
    hidden: int,
    n_heads: int,
    head_size: int,
) -> dict[str, torch.Tensor]:
    """Split HF ``c_attn`` into HF-layout q/k/v/o tensors."""
    w = attn.c_attn.weight.detach()
    q_hf = w[:, :hidden].reshape(hidden, n_heads, head_size)
    k_hf = w[:, hidden : 2 * hidden].reshape(hidden, n_heads, head_size)
    v_hf = w[:, 2 * hidden : 3 * hidden].reshape(hidden, n_heads, head_size)

    o_w = attn.c_proj.weight.detach().reshape(n_heads, head_size, hidden)

    bias = attn.c_attn.bias.detach()
    q_b = bias[:hidden].reshape(n_heads, head_size).transpose(0, 1).contiguous()
    k_b = bias[hidden : 2 * hidden].reshape(n_heads, head_size).transpose(0, 1).contiguous()
    v_b = bias[2 * hidden : 3 * hidden].reshape(n_heads, head_size).transpose(0, 1).contiguous()

    return {
        "q_weight": torch_to_nntile_qkv_weight(q_hf),
        "k_weight": torch_to_nntile_qkv_weight(k_hf),
        "v_weight": torch_to_nntile_qkv_weight(v_hf),
        "o_weight": torch_to_nntile_o_weight(o_w),
        "q_bias": q_b.contiguous(),
        "k_bias": k_b.contiguous(),
        "v_bias": v_b.contiguous(),
        "o_bias": attn.c_proj.bias.detach().contiguous(),
    }


def load_hf_into_gpt2_lm_head(
    minimal: GPT2LMHead,
    hf: GPT2LMHeadModel,
) -> None:
    """Copy HF weights into ``minimal`` (CPU tensors; call ``.to('nntile')`` after)."""
    tr = minimal.transformer
    hf_tr = hf.transformer

    tr.wte.weight.data.copy_(hf_tr.wte.weight.data)
    tr.wpe.weight.data.copy_(hf_tr.wpe.weight.data)
    tr.ln_f.weight.data.copy_(hf_tr.ln_f.weight.data)
    tr.ln_f.bias.data.copy_(hf_tr.ln_f.bias.data)

    hidden = minimal.config.n_embd
    n_heads = minimal.config.n_head
    head_size = hidden // n_heads

    for block, hf_block in zip(tr.h, hf_tr.h):
        block.ln_1.weight.data.copy_(hf_block.ln_1.weight.data)
        block.ln_1.bias.data.copy_(hf_block.ln_1.bias.data)
        block.ln_2.weight.data.copy_(hf_block.ln_2.weight.data)
        block.ln_2.bias.data.copy_(hf_block.ln_2.bias.data)

        attn_w = _split_hf_attn_weights(
            hf_block.attn,
            hidden,
            n_heads,
            head_size,
        )
        block.attn.q_weight.data.copy_(attn_w["q_weight"])
        block.attn.k_weight.data.copy_(attn_w["k_weight"])
        block.attn.v_weight.data.copy_(attn_w["v_weight"])
        block.attn.o_weight.data.copy_(attn_w["o_weight"])
        block.attn.q_bias.data.copy_(attn_w["q_bias"])
        block.attn.k_bias.data.copy_(attn_w["k_bias"])
        block.attn.v_bias.data.copy_(attn_w["v_bias"])
        block.attn.o_bias.data.copy_(attn_w["o_bias"])

        block.mlp.c_fc.weight.data.copy_(
            _conv1d_to_linear_weight(hf_block.mlp.c_fc.weight.data)
        )
        block.mlp.c_fc.bias.data.copy_(hf_block.mlp.c_fc.bias.data)
        block.mlp.c_proj.weight.data.copy_(
            _conv1d_to_linear_weight(hf_block.mlp.c_proj.weight.data)
        )
        block.mlp.c_proj.bias.data.copy_(hf_block.mlp.c_proj.bias.data)

    if minimal.config.tie_word_embeddings:
        minimal.tie_weights()
    else:
        minimal.lm_head.weight.data.copy_(hf.lm_head.weight.data.contiguous())


__all__ = ["load_hf_into_gpt2_lm_head"]
