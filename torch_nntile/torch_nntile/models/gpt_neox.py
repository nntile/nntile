# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/gpt_neox.py
# GPT-NeoX causal LM for device="nntile".

"""GPT-NeoX stack mirroring ``nntile::model::gptneox`` (RoPE, parallel residual)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_nntile.models.gpt2_minimal import make_causal_sdpa_mask

try:
    from torch_nntile.rope import rope, rope_sin_cos_from_position_ids
except ImportError:  # pragma: no cover
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


@dataclass
class GPTNeoXConfig:
    vocab_size: int = 50280
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    max_position_embeddings: int = 2048
    head_dim: int = 64
    layer_norm_eps: float = 1e-5
    rotary_pct: float = 0.25
    rotary_emb_base: float = 10000.0
    use_parallel_residual: bool = True
    attention_bias: bool = True
    tie_word_embeddings: bool = False
    attention_layers: list[str] = field(default_factory=list)
    eos_token_id: int = 50256
    bos_token_id: int = 50256
    name: str = "gpt-neox"

    def __post_init__(self) -> None:
        if (
            self.num_attention_heads > 0
            and self.hidden_size % self.num_attention_heads == 0
        ):
            self.head_dim = self.hidden_size // self.num_attention_heads

    def validate(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "GPTNeoXConfig: hidden_size must be divisible by "
                "num_attention_heads"
            )

    @property
    def rotary_ndims(self) -> int:
        return int(self.head_dim * self.rotary_pct)


class GPTNeoXAttention(nn.Module):
    def __init__(self, config: GPTNeoXConfig) -> None:
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden = config.hidden_size
        self.rotary_ndims = config.rotary_ndims
        bias = config.attention_bias
        self.query_key_value = nn.Linear(
            self.hidden, 3 * self.hidden, bias=bias
        )
        self.dense = nn.Linear(self.hidden, self.hidden, bias=bias)

    def _apply_rope(self, x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
        # Partial RoPE: rotate first rotary_ndims dims, pass through the rest.
        rot = self.rotary_ndims
        if rot <= 0:
            return x
        x_rot, x_pass = x[..., :rot], x[..., rot:]
        sin_h = sin.unsqueeze(1)
        cos_h = cos.unsqueeze(1)
        # sin/cos sized for rotary_ndims // 2
        x_rot = rope(sin_h, cos_h, x_rot)
        return torch.cat([x_rot, x_pass], dim=-1)

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
        qkv = self.query_key_value(x)
        qkv = qkv.view(b, s, self.n_heads, 3 * self.head_dim)
        q, k, v = qkv.split(self.head_dim, dim=-1)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if sin is not None and cos is not None:
            q = self._apply_rope(q, sin, cos)
            k = self._apply_rope(k, sin, cos)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal and attn_mask is None,
        )
        out = out.transpose(1, 2).contiguous().view(b, s, self.hidden)
        return self.dense(out)


class GPTNeoXMLP(nn.Module):
    def __init__(self, config: GPTNeoXConfig) -> None:
        super().__init__()
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size, config.intermediate_size
        )
        self.dense_4h_to_h = nn.Linear(
            config.intermediate_size, config.hidden_size
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.dense_4h_to_h(self.act(self.dense_h_to_4h(x)))


class GPTNeoXLayer(nn.Module):
    def __init__(self, config: GPTNeoXConfig) -> None:
        super().__init__()
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.attention = GPTNeoXAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.mlp = GPTNeoXMLP(config)

    def forward(
        self,
        x: Tensor,
        sin: Tensor | None = None,
        cos: Tensor | None = None,
        attn_mask: Tensor | None = None,
        *,
        is_causal: bool = True,
    ) -> Tensor:
        if self.use_parallel_residual:
            # attn and mlp from the same normalized input, then sum.
            normed = self.input_layernorm(x)
            attn_out = self.attention(
                normed, sin, cos, attn_mask, is_causal=is_causal
            )
            mlp_out = self.mlp(self.post_attention_layernorm(x))
            return x + attn_out + mlp_out
        residual = x
        x = self.input_layernorm(x)
        x = self.attention(x, sin, cos, attn_mask, is_causal=is_causal)
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x


class GPTNeoXModel(nn.Module):
    def __init__(self, config: GPTNeoXConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [GPTNeoXLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

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
            )
            if input_ids.device.type != "cpu":
                position_ids = position_ids.to(input_ids.device)
        rotary_dim = self.config.rotary_ndims
        if (sin is None or cos is None) and rotary_dim > 0:
            sin, cos = rope_sin_cos_from_position_ids(
                position_ids,
                rotary_dim,
                rope_theta=self.config.rotary_emb_base,
            )
        x = self.embed_in(input_ids)
        for layer in self.layers:
            x = layer(x, sin, cos, attn_mask, is_causal=is_causal)
        return self.final_layer_norm(x)


class GPTNeoXCausal(nn.Module):
    """GPT-NeoX causal LM (``nntile::model::gptneox::GptneoxCausal``)."""

    def __init__(self, config: GPTNeoXConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        if config.tie_word_embeddings:
            self.embed_out.weight = self.gpt_neox.embed_in.weight

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        sin: Tensor | None = None,
        cos: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        hidden = self.gpt_neox(
            input_ids,
            position_ids=position_ids,
            sin=sin,
            cos=cos,
            attn_mask=attn_mask,
            is_causal=True,
        )
        return self.embed_out(hidden)


__all__ = [
    "GPTNeoXAttention",
    "GPTNeoXCausal",
    "GPTNeoXConfig",
    "GPTNeoXLayer",
    "GPTNeoXMLP",
    "GPTNeoXModel",
    "make_causal_sdpa_mask",
]
