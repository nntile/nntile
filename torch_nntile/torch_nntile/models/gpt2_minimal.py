# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/gpt2_minimal.py
# Minimal GPT-2 model for device="nntile" (NNTile attention weight layouts).

"""Minimal GPT-2 stack mirroring ``nntile::model::gpt2`` graph modules.

Uses the same NNTile primitives as the C++ model (``gemm``, ``transpose``,
``add_fiber``, ``sdpa_kernel``, classic ``LayerNorm`` / ``Embedding``,
``gelutanh``, residual ``add``) - not broadcast ``scale_slice``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from transformers import GPT2Config

from torch_nntile.add_fiber import add_fiber
from torch_nntile.gemm import gemm
from torch_nntile.nn import Embedding, GELU, LayerNorm
from torch_nntile.nn.functional import add
from torch_nntile.nn.sdpa import make_causal_sdpa_mask
from torch_nntile.nn.sdpa import nntile_model_transpose, sdpa_kernel


from torch_nntile.nn.sdpa import make_causal_sdpa_mask


class NntileConv1D(nn.Module):
    """HF ``Conv1D`` equivalent via NNTile ``gemm`` + bias (no ``weight.t()``)."""

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
        # PyTorch Linear layout ``(out_features, in_features)``.
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
        # ``[..., in] @ [out, in]`` with ``trans_b=True`` (NNTile gemm flag).
        out = gemm(
            x,
            self.weight,
            ndim=1,
            batch_ndim=0,
            trans_a=False,
            trans_b=True,
        )
        if self.bias is not None:
            # C++ ``Linear`` / ``add_fiber`` on last axis (no broadcast expand).
            out = add_fiber(self.bias, out, axis=out.dim() - 1, batch_ndim=0)
        return out


class GPT2Attention(nn.Module):
    """GPT-2 attention with NNTile-layout Q/K/V/O weights (C++ ``Gpt2Attention``)."""

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

        # C++ ``Gpt2Attention``: biases are ``(n_heads, head_size)``, applied
        # via ``add_fiber`` after ``transpose(1)`` into SDPA kernel layout.
        self.q_bias = nn.Parameter(torch.zeros(n_heads, hs))
        self.k_bias = nn.Parameter(torch.zeros(n_heads, hs))
        self.v_bias = nn.Parameter(torch.zeros(n_heads, hs))
        self.o_bias = nn.Parameter(torch.zeros(hidden))

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

    def _qkv_proj(self, x: Tensor, weight: Tensor) -> Tensor:
        """``gemm(x, w, ndim=1)`` -> ``[batch, seq, head_size, n_heads]``."""
        return gemm(x, weight, ndim=1, batch_ndim=0)

    def _add_qkv_bias(self, x_sdpa: Tensor, bias: Tensor) -> Tensor:
        """Add ``(n_heads, head_size)`` bias via ``add_fiber`` (C++ axis=3, batch=1)."""
        # ``x_sdpa``: ``[n_heads, batch, seq, head_size]``
        return add_fiber(bias, x_sdpa, axis=3, batch_ndim=1)

    def _output_proj(self, attn_out: Tensor) -> Tensor:
        """``gemm(attn, w_o, ndim=2)`` + ``o_bias`` via ``add_fiber``."""
        out = gemm(attn_out, self.o_weight, ndim=2, batch_ndim=0)
        return add_fiber(self.o_bias, out, axis=out.dim() - 1, batch_ndim=0)

    def forward(self, x: Tensor, causal_mask: Tensor | None) -> Tensor:
        # Mirror ``nntile/src/model/gpt2/gpt2_attention.cc``:
        # gemm -> transpose(1) -> add_fiber(bias) -> sdpa_eager -> transpose(3) -> O.
        q = nntile_model_transpose(self._qkv_proj(x, self.q_weight), 1)
        q = self._add_qkv_bias(q, self.q_bias)
        k = nntile_model_transpose(self._qkv_proj(x, self.k_weight), 1)
        k = self._add_qkv_bias(k, self.k_bias)
        v = nntile_model_transpose(self._qkv_proj(x, self.v_weight), 1)
        v = self._add_qkv_bias(v, self.v_bias)
        attn_out = sdpa_kernel(q, k, v, mask=causal_mask, batch_ndim=2)
        attn_out = nntile_model_transpose(attn_out, 3)
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
        self.act = GELU(approximate="tanh")

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        return self.c_proj(x)


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(self, x: Tensor, causal_mask: Tensor | None) -> Tensor:
        residual = x
        x = self.ln_1(x)
        x = self.attn(x, causal_mask)
        x = add(residual, x)
        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        return add(residual, x)


class GPT2Model(nn.Module):
    """GPT-2 transformer backbone (token + position embeddings, blocks, ln_f)."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        self.wte = Embedding(config.vocab_size, config.n_embd)
        self.wpe = Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList(
            [GPT2Block(config) for _ in range(config.n_layer)]
        )
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        # Built once on CPU, moved to the request device, then reused.
        # Keys: position_ids -> (batch, seq); causal_mask -> seq.
        self._position_ids_cache: dict[tuple[int, int], Tensor] = {}
        self._causal_mask_cache: dict[int, Tensor] = {}

    def _cached_position_ids(self, input_ids: Tensor) -> Tensor:
        """Return ``[batch, seq]`` position ids on ``input_ids.device`` (cached)."""
        batch, seq = int(input_ids.size(0)), int(input_ids.size(-1))
        key = (batch, seq)
        cached = self._position_ids_cache.get(key)
        if cached is not None and cached.device == input_ids.device:
            return cached
        position_ids = (
            torch.arange(seq, dtype=torch.long, device="cpu")
            .unsqueeze(0)
            .expand(batch, seq)
            .contiguous()
        )
        if input_ids.device.type != "cpu":
            position_ids = position_ids.to(input_ids.device)
        self._position_ids_cache[key] = position_ids
        return position_ids

    def _cached_causal_mask(self, input_ids: Tensor) -> Tensor:
        """Return ``[seq, seq]`` BOOL causal mask on ``input_ids.device`` (cached)."""
        seq = int(input_ids.size(-1))
        cached = self._causal_mask_cache.get(seq)
        if cached is not None and cached.device == input_ids.device:
            return cached
        causal_mask = make_causal_sdpa_mask(seq, device=input_ids.device)
        self._causal_mask_cache[seq] = causal_mask
        return causal_mask

    def clear_sequence_caches(self) -> None:
        """Drop cached position_ids / causal masks (e.g. after device change)."""
        self._position_ids_cache.clear()
        self._causal_mask_cache.clear()

    def warm_sequence_caches(
        self,
        *,
        batch_sizes: list[int] | tuple[int, ...],
        seq_len: int,
        device: torch.device | str,
    ) -> None:
        """Build position_ids / causal mask once on ``device`` for training reuse."""
        device = torch.device(device)
        for batch in sorted({int(b) for b in batch_sizes}):
            if batch < 1:
                raise ValueError(f"batch size must be >= 1, got {batch}")
            probe = torch.empty(
                (batch, seq_len),
                dtype=torch.long,
                device=device,
            )
            self._cached_position_ids(probe)
            self._cached_causal_mask(probe)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        causal_mask: Tensor | None = None,
    ) -> Tensor:
        if position_ids is None:
            position_ids = self._cached_position_ids(input_ids)
        if causal_mask is None:
            causal_mask = self._cached_causal_mask(input_ids)

        x = add(self.wte(input_ids), self.wpe(position_ids))
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
        # Weight tying intentionally unsupported (independent lm_head).
        # Config.tie_word_embeddings is ignored for now (migration debt).
        return

    def _tie_weights(self) -> None:
        # Kept for API compatibility; does not share storage.
        return

    def tie_weights(self) -> None:
        """No-op: embedding/lm_head tying is deferred (migration debt)."""
        return

    def clear_sequence_caches(self) -> None:
        self.transformer.clear_sequence_caches()

    def warm_sequence_caches(
        self,
        *,
        batch_sizes: list[int] | tuple[int, ...],
        seq_len: int,
        device: torch.device | str,
    ) -> None:
        """Prefetch cached position_ids / causal masks onto ``device``."""
        self.transformer.warm_sequence_caches(
            batch_sizes=batch_sizes,
            seq_len=seq_len,
            device=device,
        )

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
