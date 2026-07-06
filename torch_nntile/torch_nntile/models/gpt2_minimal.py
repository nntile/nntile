# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/gpt2_minimal.py
# Minimal GPT-2 model for device="nntile" (NNTile attention weight layouts).

"""Minimal GPT-2 stack mirroring ``nntile::model::gpt2`` graph modules."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from transformers import GPT2Config

from torch_nntile.nn import SDPA
from torch_nntile.gemm import gemm


def make_causal_sdpa_mask(seq_len: int, device: torch.device | None = None) -> Tensor:
    """BOOL causal mask ``[seq, seq]`` with ``mask[q, k] = (k <= q)``."""
    q_idx = torch.arange(seq_len, device=device)
    k_idx = torch.arange(seq_len, device=device)
    return k_idx.unsqueeze(0) <= q_idx.unsqueeze(1)


class NntileConv1D(nn.Module):
    """HF ``Conv1D`` equivalent via ``mm`` + bias (autograd-friendly on nntile)."""

    def __init__(
        self,
        nf: int,
        nx: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.weight = nn.Parameter(torch.empty(nf, nx))
        if bias:
            self.bias = nn.Parameter(torch.zeros(nf))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        x2d = x.reshape(-1, x.size(-1))
        out = torch.mm(x2d, self.weight.t())
        if self.bias is not None:
            bias_bc = (
                self.bias.view(1, -1)
                .expand(x2d.size(0), -1)
                .contiguous()
            )
            out = out + bias_bc
        return out.view(size_out)


class GPT2Attention(nn.Module):
    """GPT-2 attention with NNTile-layout Q/K/V/O weights and ``SDPA``."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.hidden = config.n_embd

        hs = self.head_size
        n_heads = self.n_head
        hidden = self.hidden

        self.q_weight = nn.Parameter(torch.empty(hidden, hs, n_heads))
        self.k_weight = nn.Parameter(torch.empty(hidden, hs, n_heads))
        self.v_weight = nn.Parameter(torch.empty(hidden, hs, n_heads))
        self.o_weight = nn.Parameter(torch.empty(hs, n_heads, hidden))

        self.q_bias = nn.Parameter(torch.zeros(n_heads, hs))
        self.k_bias = nn.Parameter(torch.zeros(n_heads, hs))
        self.v_bias = nn.Parameter(torch.zeros(n_heads, hs))
        self.o_bias = nn.Parameter(torch.zeros(hidden))

        self.sdpa = SDPA(batch_ndim=2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in (
            self.q_weight,
            self.k_weight,
            self.v_weight,
            self.o_weight,
        ):
            nn.init.normal_(p, std=0.02)
        for p in (self.q_bias, self.k_bias, self.v_bias, self.o_bias):
            nn.init.zeros_(p)

    def _project(
        self,
        x: Tensor,
        weight: Tensor,
        bias: Tensor,
    ) -> Tensor:
        """``gemm(x, w, ndim=1)`` + bias → ``[batch, seq, head_size, n_heads]``."""
        bsz, seq, head_size, n_heads = (
            x.size(0),
            x.size(1),
            weight.size(1),
            weight.size(2),
        )
        proj = gemm(x, weight, ndim=1, batch_ndim=0)
        bias_bc = (
            bias.transpose(0, 1)
            .view(1, 1, head_size, n_heads)
            .expand(bsz, seq, head_size, n_heads)
            .contiguous()
        )
        return proj + bias_bc

    def _output_proj(self, attn_out: Tensor) -> Tensor:
        """``gemm(attn, w_o, ndim=2)`` + bias on post-SDPA projection layout."""
        bsz, seq = attn_out.size(0), attn_out.size(1)
        hidden = self.hidden
        out = gemm(attn_out, self.o_weight, ndim=2, batch_ndim=0)
        bias_bc = (
            self.o_bias.view(1, 1, hidden)
            .expand(bsz, seq, hidden)
            .contiguous()
        )
        return out + bias_bc

    def forward(self, x: Tensor, causal_mask: Tensor | None) -> Tensor:
        q = self._project(x, self.q_weight, self.q_bias)
        k = self._project(x, self.k_weight, self.k_bias)
        v = self._project(x, self.v_weight, self.v_bias)
        attn_out = self.sdpa(q, k, v, mask=causal_mask)
        return self._output_proj(attn_out)


class GPT2MLP(nn.Module):
    """GPT-2 MLP (GELU tanh approx) using ``NntileConv1D``."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        inner = config.n_inner
        if inner is None:
            inner = 4 * config.n_embd
        self.c_fc = NntileConv1D(inner, config.n_embd)
        self.c_proj = NntileConv1D(config.n_embd, inner)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        return self.c_proj(x)


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(self, x: Tensor, causal_mask: Tensor | None) -> Tensor:
        residual = x
        x = self.ln_1(x)
        x = self.attn(x, causal_mask)
        x = residual + x
        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        return residual + x


class GPT2Model(nn.Module):
    """GPT-2 transformer backbone (token + position embeddings, blocks, ln_f)."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList(
            [GPT2Block(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        causal_mask: Tensor | None = None,
    ) -> Tensor:
        if position_ids is None:
            seq = input_ids.size(-1)
            position_ids = (
                torch.arange(seq, dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand(input_ids.size(0), -1)
                .contiguous()
            )

        if causal_mask is None:
            causal_mask = make_causal_sdpa_mask(
                input_ids.size(-1),
                device=input_ids.device,
            )

        x = self.wte(input_ids) + self.wpe(position_ids)
        for block in self.h:
            x = block(x, causal_mask)
        return self.ln_f(x)


class GPT2LMHead(nn.Module):
    """GPT-2 causal LM (``Gpt2Causal`` / ``GPT2LMHeadModel`` equivalent)."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = NntileConv1D(
            config.vocab_size,
            config.n_embd,
            bias=False,
        )
        self.post_init()

    def post_init(self) -> None:
        if self.config.tie_word_embeddings:
            self._tie_weights()

    def _tie_weights(self) -> None:
        self.lm_head.weight = self.transformer.wte.weight  # share storage

    def tie_weights(self) -> None:
        """Re-apply word embedding tie (call after ``.to('nntile')``)."""
        if self.config.tie_word_embeddings:
            self._tie_weights()

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        causal_mask: Tensor | None = None,
    ) -> Tensor:
        hidden = self.transformer(input_ids, position_ids, causal_mask)
        return self.lm_head(hidden)


__all__ = [
    "GPT2Attention",
    "GPT2Block",
    "GPT2LMHead",
    "GPT2MLP",
    "GPT2Model",
    "NntileConv1D",
    "make_causal_sdpa_mask",
]
