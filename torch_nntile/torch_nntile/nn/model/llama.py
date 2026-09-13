# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/nn/model/llama.py
# Llama causal LM for device="nntile".

"""Llama stack mirroring ``nntile::model::llama`` (RMSNorm, RoPE, SiLU MLP).

Forward / backward keep activations on ``device=nntile`` end-to-end.
RoPE uses ``arange(seq)`` for every batch row. ``sin``/``cos`` are
``[seq, head_dim // 2]``; extra leading modes of ``x`` (heads, GQA,
batch) are the kernel batch ``n``. There is no ``[B, S]`` position
table.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from torch_nntile import _C
from torch_nntile.gemm import gemm
from torch_nntile.nn import Embedding, SiLU
from torch_nntile.nn.functional import add, mul
from torch_nntile.nn.linear import NntileLinear, prepare_sdpa_mask
from torch_nntile.nn.model.gpt2_minimal import make_causal_sdpa_mask
from torch_nntile.nn.sdpa import nntile_model_transpose, sdpa_kernel
from torch_nntile.normalization import rms_norm

try:
    from torch_nntile.rope import rope, rope_sin_cos_from_position_ids
except ImportError:  # pragma: no cover - stub if rope.py missing
    def rope(sin: Tensor, cos: Tensor, x: Tensor) -> Tensor:
        return x

    def rope_sin_cos_from_position_ids(
        position_ids: Tensor,
        head_dim: int,
        *,
        rope_theta: float = 10000.0,
    ) -> tuple[Tensor, Tensor]:
        del rope_theta
        half = head_dim // 2
        shape = (*tuple(position_ids.shape), half)
        z = torch.zeros(
            shape, dtype=torch.float32, device=position_ids.device
        )
        return z, torch.ones_like(z)


def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """GQA KV expand via classic ``scale_slice``, not ``unsqueeze``+``repeat``.

    ``aten::unsqueeze`` is a view that keeps the 4-D storage TensorRef, so
    SDPA backward can hand ``rope_backward`` a 5-D node vs 4-D ``dx``.
    Matches C++ ``repeat_kv_heads`` / NNGraph ``scale_slice`` on axis 1.
    ``x`` is ``[H_kv, B, S, D]`` -> ``[H_kv, n_rep, B, S, D]``.
    """
    if n_rep == 1:
        return x
    if x.device.type == "nntile":
        return _C.scale_slice(x, 1, n_rep)
    return x.unsqueeze(1).repeat(1, n_rep, 1, 1, 1)


@dataclass
class LlamaConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    max_position_embeddings: int = 2048
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = False
    eos_token_id: int = 2
    bos_token_id: int = 1
    name: str = "llama"

    def __post_init__(self) -> None:
        if (
            self.num_attention_heads > 0
            and self.hidden_size % self.num_attention_heads == 0
        ):
            self.head_dim = self.hidden_size // self.num_attention_heads

    def validate(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "LlamaConfig: hidden_size must be divisible by "
                "num_attention_heads"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "LlamaConfig: num_attention_heads must be divisible by "
                "num_key_value_heads"
            )


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.normalized_shape = (hidden_size,)

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.normalized_shape, self.weight, self.eps)


class LlamaMLP(nn.Module):
    """Gated MLP: ``down(SiLU(gate(x)) * up(x))``."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        h = config.hidden_size
        i = config.intermediate_size
        bias = config.mlp_bias
        self.gate_proj = NntileLinear(h, i, bias=bias)
        self.up_proj = NntileLinear(h, i, bias=bias)
        self.down_proj = NntileLinear(i, h, bias=bias)
        self.act = SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(mul(self.act(self.gate_proj(x)), self.up_proj(x)))


class LlamaAttention(nn.Module):
    """Multi-head attention with optional GQA and RoPE."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.use_gqa = self.n_rep > 1
        if self.use_gqa:
            self.q_weight = nn.Parameter(
                torch.empty(
                    self.hidden_size,
                    self.head_dim,
                    self.n_kv_heads,
                    self.n_rep,
                )
            )
            self.o_weight = nn.Parameter(
                torch.empty(
                    self.head_dim,
                    self.n_kv_heads,
                    self.n_rep,
                    self.hidden_size,
                )
            )
        else:
            self.q_weight = nn.Parameter(
                torch.empty(self.hidden_size, self.head_dim, self.n_heads)
            )
            self.o_weight = nn.Parameter(
                torch.empty(self.head_dim, self.n_heads, self.hidden_size)
            )
        self.k_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.head_dim, self.n_kv_heads)
        )
        self.v_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.head_dim, self.n_kv_heads)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in (
            self.q_weight,
            self.k_weight,
            self.v_weight,
            self.o_weight,
        ):
            nn.init.normal_(p, std=0.02)

    def _project_q(self, x: Tensor) -> Tensor:
        q = gemm(x, self.q_weight, ndim=1, batch_ndim=0)
        return nntile_model_transpose(q, 2 if self.use_gqa else 1)

    def _project_kv(self, x: Tensor, weight: Tensor) -> Tensor:
        out = gemm(x, weight, ndim=1, batch_ndim=0)
        return nntile_model_transpose(out, 1)

    def _output(self, attn_out: Tensor) -> Tensor:
        attn_t = nntile_model_transpose(attn_out, 3)
        out_ndim = 3 if self.use_gqa else 2
        return gemm(attn_t, self.o_weight, ndim=out_ndim, batch_ndim=0)

    def _apply_rope(self, x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
        # sin/cos are ``[S, head_dim // 2]``. Extra leading modes of
        # ``x`` (heads, GQA n_rep, batch) are the kernel batch ``n``.
        if sin.device != x.device:
            sin = sin.to(x.device)
            cos = cos.to(x.device)
        return rope(sin, cos, x)

    def forward(
        self,
        x: Tensor,
        sin: Tensor | None = None,
        cos: Tensor | None = None,
        attn_mask: Tensor | None = None,
        *,
        is_causal: bool = True,
    ) -> Tensor:
        s = int(x.size(1))
        q = self._project_q(x)
        k = self._project_kv(x, self.k_weight)
        v = self._project_kv(x, self.v_weight)
        if sin is not None and cos is not None:
            q = self._apply_rope(q, sin, cos)
            k = self._apply_rope(k, sin, cos)
        batch_ndim = 2
        if self.use_gqa:
            k = _repeat_kv(k, self.n_rep)
            v = _repeat_kv(v, self.n_rep)
            batch_ndim = 3
        mask = attn_mask
        if mask is None and is_causal:
            mask = prepare_sdpa_mask(
                None,
                x,
                q_len=s,
                is_causal=True,
            )
        out = sdpa_kernel(
            q,
            k,
            v,
            mask=mask,
            batch_ndim=batch_ndim,
        )
        return self._output(out)


class LlamaDecoder(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        x: Tensor,
        sin: Tensor | None = None,
        cos: Tensor | None = None,
        attn_mask: Tensor | None = None,
        *,
        is_causal: bool = True,
    ) -> Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, sin, cos, attn_mask, is_causal=is_causal)
        x = add(residual, x)
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return add(residual, x)


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoder(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Host-built RoPE / causal-mask tables (NNGraph bind_data).
        # RoPE is ``[seq, head_dim // 2]``; batch is kernel n, not a table.
        self._rope_cache: dict[int, tuple[Tensor, Tensor]] = {}
        self._causal_mask_cache: dict[int, Tensor] = {}

    def _cached_rope(
        self, seq: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Return ``[seq, head_dim // 2]`` sin/cos (built once, then reused).

        Built from ``arange(seq)``. The RoPE kernel applies that table
        across every batch row.
        """
        cached = self._rope_cache.get(seq)
        if cached is not None and cached[0].device == device:
            return cached
        pos_host = torch.arange(seq, dtype=torch.long, device="cpu")
        sin, cos = rope_sin_cos_from_position_ids(
            pos_host,
            self.config.head_dim,
            rope_theta=self.config.rope_theta,
        )
        if device.type != "cpu":
            sin = sin.to(device)
            cos = cos.to(device)
        self._rope_cache[seq] = (sin, cos)
        return sin, cos

    def _cached_causal_mask(
        self, seq: int, device: torch.device
    ) -> Tensor:
        """Return ``[seq, seq]`` BOOL causal mask (cached, like GPT-2)."""
        cached = self._causal_mask_cache.get(seq)
        if cached is not None and cached.device == device:
            return cached
        causal_mask = make_causal_sdpa_mask(seq, device=device)
        self._causal_mask_cache[seq] = causal_mask
        return causal_mask

    def clear_sequence_caches(self) -> None:
        self._rope_cache.clear()
        self._causal_mask_cache.clear()

    def warm_sequence_caches(
        self,
        *,
        batch_sizes: list[int] | tuple[int, ...] | None = None,
        seq_len: int,
        device: torch.device | str,
    ) -> None:
        """Prepare RoPE / causal-mask tables once for training reuse."""
        del batch_sizes
        device = torch.device(device)
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        self._cached_rope(seq_len, device)
        self._cached_causal_mask(seq_len, device)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        sin: Tensor | None = None,
        cos: Tensor | None = None,
        attn_mask: Tensor | None = None,
        *,
        is_causal: bool = True,
    ) -> Tensor:
        # Optional ``position_ids`` is HF-shaped leftover. Training is
        # ``arange(seq)`` for every batch row; B is consumed by RoPE.
        seq = int(
            position_ids.size(-1)
            if position_ids is not None
            else input_ids.size(-1)
        )
        if sin is None or cos is None:
            sin, cos = self._cached_rope(seq, input_ids.device)
        if attn_mask is None and is_causal:
            attn_mask = self._cached_causal_mask(seq, input_ids.device)
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, sin, cos, attn_mask, is_causal=is_causal)
        return self.norm(x)


class LlamaCausal(nn.Module):
    """LlamaModel + lm_head (``nntile::model::llama::LlamaCausal``)."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = NntileLinear(
            config.hidden_size, config.vocab_size, bias=False
        )
        # Weight tying intentionally unsupported (independent lm_head).

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        sin: Tensor | None = None,
        cos: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        hidden = self.model(
            input_ids,
            position_ids=position_ids,
            sin=sin,
            cos=cos,
            attn_mask=attn_mask,
            is_causal=True,
        )
        return self.lm_head(hidden)


__all__ = [
    "LlamaAttention",
    "LlamaCausal",
    "LlamaConfig",
    "LlamaDecoder",
    "LlamaMLP",
    "LlamaModel",
    "LlamaRMSNorm",
    "make_causal_sdpa_mask",
]
