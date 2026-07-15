# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/llama.py
# Llama causal LM for device="nntile".

"""Llama stack mirroring ``nntile::model::llama`` (RMSNorm, RoPE, SiLU MLP)."""

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
        z = torch.zeros(b, s, half, dtype=torch.float32, device=position_ids.device)
        return z, torch.ones_like(z)


class _RepeatKV(torch.autograd.Function):
    """``repeat_interleave`` for GQA; CPU round-trip on ``device=nntile``."""

    @staticmethod
    def forward(ctx, x: Tensor, n_rep: int) -> Tensor:  # type: ignore[override]
        ctx.n_rep = int(n_rep)
        if n_rep == 1:
            return x
        if x.device.type != "nntile":
            return x.repeat_interleave(n_rep, dim=1)
        y = (
            x.detach()
            .to("cpu")
            .repeat_interleave(n_rep, dim=1)
            .contiguous()
        )
        return y.to(x.device)

    @staticmethod
    def backward(ctx, grad_y: Tensor):  # type: ignore[override]
        n_rep = ctx.n_rep
        if n_rep == 1:
            return grad_y, None
        g = grad_y.detach().to("cpu")
        b, h, s, d = g.shape
        g = g.view(b, h // n_rep, n_rep, s, d).sum(dim=2).contiguous()
        if grad_y.device.type != "cpu":
            g = g.to(grad_y.device)
        return g, None


def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    return _RepeatKV.apply(x, n_rep)


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
        # x: [B, H, S, D]; sin/cos: [B, S, D/2] or already [B, H, S, D/2].
        # Expand on CPU — nntile rejects non-contiguous expand views.
        n_heads = x.size(1)
        if sin.dim() == 3:
            sin_c = sin.detach().to("cpu")
            cos_c = cos.detach().to("cpu")
            sin_h = (
                sin_c.unsqueeze(1)
                .expand(-1, n_heads, -1, -1)
                .contiguous()
            )
            cos_h = (
                cos_c.unsqueeze(1)
                .expand(-1, n_heads, -1, -1)
                .contiguous()
            )
            if x.device.type != "cpu":
                sin_h = sin_h.to(x.device)
                cos_h = cos_h.to(x.device)
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
        b, s = input_ids.shape
        if position_ids is None:
            position_ids = (
                torch.arange(s, dtype=torch.long, device="cpu")
                .unsqueeze(0)
                .expand(b, s)
                .contiguous()
            )
            if input_ids.device.type != "cpu":
                position_ids = position_ids.to(input_ids.device)
        if sin is None or cos is None:
            # Keep RoPE tables on CPU; attention expands/moves per head count.
            pos_cpu = position_ids.detach().to("cpu").contiguous()
            sin, cos = rope_sin_cos_from_position_ids(
                pos_cpu,
                self.config.head_dim,
                rope_theta=self.config.rope_theta,
            )
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
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

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
