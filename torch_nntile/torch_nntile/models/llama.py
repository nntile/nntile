# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/llama.py
# Llama causal LM for device="nntile".

"""Llama stack mirroring ``nntile::model::llama`` (RMSNorm, RoPE, SiLU MLP).

Forward / backward keep activations on ``device=nntile`` end-to-end.
``position_ids`` / RoPE ``sin``/``cos`` are one-shot host tables (see
``warm_sequence_caches``), matching deleted NNGraph ``bind_data`` — prepared
once for training, not recomputed from activations each step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_nntile.models.gpt2_minimal import make_causal_sdpa_mask
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
        b, s = position_ids.shape
        half = head_dim // 2
        z = torch.zeros(
            b, s, half, dtype=torch.float32, device=position_ids.device
        )
        return z, torch.ones_like(z)


def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """GQA KV expand via ``aten::repeat`` (nntile scale-slice), not host.

    Matches deleted NNGraph ``scale_slice(..., kv_group_size)``.
    ``x`` is ``[B, H_kv, S, D]`` → ``[B, H_kv * n_rep, S, D]``.
    """
    if n_rep == 1:
        return x
    b, h_kv, s, d = x.shape
    return (
        x.view(b, h_kv, 1, s, d)
        .repeat(1, 1, n_rep, 1, 1)
        .view(b, h_kv * n_rep, s, d)
    )


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
        self.gate_proj = nn.Linear(h, i, bias=bias)
        self.up_proj = nn.Linear(h, i, bias=bias)
        self.down_proj = nn.Linear(i, h, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    """Multi-head attention with optional GQA and RoPE."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        bias = config.attention_bias
        self.q_proj = nn.Linear(
            self.hidden_size, self.n_heads * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.n_kv_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.n_kv_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, self.hidden_size, bias=bias
        )

    def _shape(self, x: Tensor, n_heads: int) -> Tensor:
        b, s, _ = x.shape
        return x.view(b, s, n_heads, self.head_dim).transpose(1, 2)

    def _apply_rope(self, x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
        # x: [B, H, S, D]; sin/cos: [B, S, D/2] or [B, H, S, D/2].
        # RoPE tables must share x's device (upload once if needed).
        if sin.device != x.device:
            sin = sin.to(x.device)
            cos = cos.to(x.device)
        n_heads = x.size(1)
        if sin.dim() == 3:
            b, s, half = sin.shape
            sin_h = sin.view(b, 1, s, half).repeat(1, n_heads, 1, 1)
            cos_h = cos.view(b, 1, s, half).repeat(1, n_heads, 1, 1)
        else:
            sin_h = sin
            cos_h = cos
        return rope(sin_h, cos_h, x)

    def forward(
        self,
        x: Tensor,
        sin: Tensor | None = None,
        cos: Tensor | None = None,
        attn_mask: Tensor | None = None,
        *,
        is_causal: bool = True,
    ) -> Tensor:
        b, s, _ = x.shape
        q = self._shape(self.q_proj(x), self.n_heads)
        k = self._shape(self.k_proj(x), self.n_kv_heads)
        v = self._shape(self.v_proj(x), self.n_kv_heads)
        if sin is not None and cos is not None:
            q = self._apply_rope(q, sin, cos)
            k = self._apply_rope(k, sin, cos)
        if self.n_rep > 1:
            k = _repeat_kv(k, self.n_rep)
            v = _repeat_kv(v, self.n_rep)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal and attn_mask is None,
        )
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.o_proj(out)


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
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoder(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Host-built index / RoPE tables uploaded once (NNGraph bind_data).
        self._position_ids_cache: dict[tuple[int, int], Tensor] = {}
        self._rope_cache: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}

    def _cached_position_ids(self, input_ids: Tensor) -> Tensor:
        batch, seq = int(input_ids.size(0)), int(input_ids.size(-1))
        key = (batch, seq)
        cached = self._position_ids_cache.get(key)
        if cached is not None and cached.device == input_ids.device:
            return cached
        # nntile lacks aten::arange; build on host, upload once.
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

    def _cached_rope(
        self, position_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return sin/cos on ``position_ids.device`` (built once, then reused).

        Matches deleted NNGraph: RoPE tables are prepared on the host and
        bound once for training — never recomputed from activations.
        """
        batch, seq = int(position_ids.size(0)), int(position_ids.size(-1))
        key = (batch, seq)
        cached = self._rope_cache.get(key)
        if cached is not None and cached[0].device == position_ids.device:
            return cached
        # One-shot host table from arange (do not gather nntile position_ids).
        pos_host = (
            torch.arange(seq, dtype=torch.long, device="cpu")
            .unsqueeze(0)
            .expand(batch, seq)
            .contiguous()
        )
        sin, cos = rope_sin_cos_from_position_ids(
            pos_host,
            self.config.head_dim,
            rope_theta=self.config.rope_theta,
        )
        if position_ids.device.type != "cpu":
            sin = sin.to(position_ids.device)
            cos = cos.to(position_ids.device)
        self._rope_cache[key] = (sin, cos)
        return sin, cos

    def clear_sequence_caches(self) -> None:
        self._position_ids_cache.clear()
        self._rope_cache.clear()

    def warm_sequence_caches(
        self,
        *,
        batch_sizes: list[int] | tuple[int, ...],
        seq_len: int,
        device: torch.device | str,
    ) -> None:
        """Prepare position_ids / RoPE tables once for training reuse."""
        device = torch.device(device)
        for batch in sorted({int(b) for b in batch_sizes}):
            if batch < 1:
                raise ValueError(f"batch size must be >= 1, got {batch}")
            probe = torch.empty(
                (batch, seq_len), dtype=torch.long, device=device
            )
            pos = self._cached_position_ids(probe)
            self._cached_rope(pos)

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
        if position_ids is None:
            position_ids = self._cached_position_ids(input_ids)
        if sin is None or cos is None:
            sin, cos = self._cached_rope(position_ids)
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
        self.lm_head = nn.Linear(
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
