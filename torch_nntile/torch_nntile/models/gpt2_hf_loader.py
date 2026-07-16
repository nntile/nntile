# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/gpt2_hf_loader.py
# Convert weights between HuggingFace GPT-2 and minimal torch_nntile GPT2LMHead.

"""Bidirectional HF <-> NNTile-layout weight conversion for minimal GPT-2."""

from __future__ import annotations

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from torch_nntile.nn.weight_layout import (
    nntile_to_torch_o_weight,
    nntile_to_torch_qkv_weight,
    torch_to_nntile_o_weight,
    torch_to_nntile_qkv_weight,
)
from torch_nntile.models.gpt2_minimal import GPT2LMHead


def _conv1d_to_linear_weight(weight: torch.Tensor) -> torch.Tensor:
    """HF Conv1D ``(in, out)`` -> addmm ``(out, in)``."""
    return weight.t().contiguous()


def _linear_to_conv1d_weight(weight: torch.Tensor) -> torch.Tensor:
    """addmm ``(out, in)`` -> HF Conv1D ``(in, out)``."""
    return weight.t().contiguous()


def _split_hf_attn_weights(
    attn,
    hidden: int,
    n_heads: int,
    head_size: int,
) -> dict[str, torch.Tensor]:
    """Split HF ``c_attn`` into NNTile-layout q/k/v/o tensors."""
    w = attn.c_attn.weight.detach()
    q_hf = w[:, :hidden].reshape(hidden, n_heads, head_size)
    k_hf = w[:, hidden : 2 * hidden].reshape(hidden, n_heads, head_size)
    v_hf = w[:, 2 * hidden : 3 * hidden].reshape(hidden, n_heads, head_size)

    o_w = attn.c_proj.weight.detach().reshape(n_heads, head_size, hidden)

    # C++ / generate_test_data layout: ``(n_heads, head_size)``.
    bias = attn.c_attn.bias.detach()
    q_b = bias[:hidden].reshape(n_heads, head_size).contiguous()
    k_b = bias[hidden : 2 * hidden].reshape(n_heads, head_size).contiguous()
    v_b = bias[2 * hidden : 3 * hidden].reshape(n_heads, head_size).contiguous()

    return {
        "q_weight": torch_to_nntile_qkv_weight(q_hf),
        "k_weight": torch_to_nntile_qkv_weight(k_hf),
        "v_weight": torch_to_nntile_qkv_weight(v_hf),
        "o_weight": torch_to_nntile_o_weight(o_w),
        "q_bias": q_b,
        "k_bias": k_b,
        "v_bias": v_b,
        "o_bias": attn.c_proj.bias.detach().contiguous(),
    }


def _merge_nntile_attn_into_hf(
    block_attn,
    hf_attn,
    *,
    hidden: int,
    n_heads: int,
    head_size: int,
) -> None:
    """Write NNTile-layout attention weights into HF ``GPT2Attention``."""
    q_hf = nntile_to_torch_qkv_weight(block_attn.q_weight.data)
    k_hf = nntile_to_torch_qkv_weight(block_attn.k_weight.data)
    v_hf = nntile_to_torch_qkv_weight(block_attn.v_weight.data)
    c_attn_w = torch.cat(
        [
            q_hf.reshape(hidden, hidden),
            k_hf.reshape(hidden, hidden),
            v_hf.reshape(hidden, hidden),
        ],
        dim=1,
    )
    c_attn_b = torch.cat(
        [
            block_attn.q_bias.data.reshape(hidden),
            block_attn.k_bias.data.reshape(hidden),
            block_attn.v_bias.data.reshape(hidden),
        ],
        dim=0,
    )
    o_torch = nntile_to_torch_o_weight(block_attn.o_weight.data)
    hf_attn.c_attn.weight.data.copy_(c_attn_w)
    hf_attn.c_attn.bias.data.copy_(c_attn_b)
    hf_attn.c_proj.weight.data.copy_(o_torch.reshape(hidden, hidden))
    hf_attn.c_proj.bias.data.copy_(block_attn.o_bias.data)


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

    # Always keep an independent lm_head (tying deferred; migration debt).
    minimal.lm_head.weight.data.copy_(hf.lm_head.weight.data.contiguous())


def export_gpt2_lm_head_to_hf_state_dict(
    minimal: GPT2LMHead,
    *,
    config: GPT2Config | None = None,
) -> dict[str, torch.Tensor]:
    """Export minimal GPT-2 CPU weights as an HF ``GPT2LMHeadModel`` state_dict.

    ``minimal`` must hold CPU tensors (e.g. after ``clone_model_weights`` applied
    via ``load_state_dict``, or a CPU copy of the module).
    """
    cfg = config if config is not None else minimal.config
    hf = GPT2LMHeadModel(cfg).float()
    hf_tr = hf.transformer
    tr = minimal.transformer

    hf_tr.wte.weight.data.copy_(tr.wte.weight.data)
    hf_tr.wpe.weight.data.copy_(tr.wpe.weight.data)
    hf_tr.ln_f.weight.data.copy_(tr.ln_f.weight.data)
    hf_tr.ln_f.bias.data.copy_(tr.ln_f.bias.data)

    hidden = int(cfg.n_embd)
    n_heads = int(cfg.n_head)
    head_size = hidden // n_heads

    for block, hf_block in zip(tr.h, hf_tr.h):
        hf_block.ln_1.weight.data.copy_(block.ln_1.weight.data)
        hf_block.ln_1.bias.data.copy_(block.ln_1.bias.data)
        hf_block.ln_2.weight.data.copy_(block.ln_2.weight.data)
        hf_block.ln_2.bias.data.copy_(block.ln_2.bias.data)

        _merge_nntile_attn_into_hf(
            block.attn,
            hf_block.attn,
            hidden=hidden,
            n_heads=n_heads,
            head_size=head_size,
        )

        hf_block.mlp.c_fc.weight.data.copy_(
            _linear_to_conv1d_weight(block.mlp.c_fc.weight.data)
        )
        hf_block.mlp.c_fc.bias.data.copy_(block.mlp.c_fc.bias.data)
        hf_block.mlp.c_proj.weight.data.copy_(
            _linear_to_conv1d_weight(block.mlp.c_proj.weight.data)
        )
        hf_block.mlp.c_proj.bias.data.copy_(block.mlp.c_proj.bias.data)

    hf.lm_head.weight.data.copy_(minimal.lm_head.weight.data.contiguous())
    # Keep lm_head untied (migration debt).

    with torch.no_grad():
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in hf.state_dict().items()
        }


__all__ = [
    "export_gpt2_lm_head_to_hf_state_dict",
    "load_hf_into_gpt2_lm_head",
]