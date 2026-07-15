# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/t5.py
# T5 encoder-decoder for device="nntile".

"""Simplified T5 mirroring ``nntile::model::t5`` (T5ForConditionalGeneration)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_nntile.normalization import rms_norm


@dataclass
class T5Config:
    vocab_size: int = 32100
    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 1024
    num_layers: int = 6
    num_decoder_layers: int = 6
    num_heads: int = 8
    relative_attention_num_buckets: int = 32
    layer_norm_epsilon: float = 1e-6
    dropout_rate: float = 0.0
    feed_forward_proj: str = "gated-gelu"
    is_gated_act: bool = True
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 1
    decoder_start_token_id: int = 0
    name: str = "t5"

    @property
    def head_dim(self) -> int:
        return self.d_kv

    @property
    def inner_dim(self) -> int:
        return self.num_heads * self.d_kv

    def validate(self) -> None:
        if self.d_model <= 0 or self.num_heads <= 0 or self.d_kv <= 0:
            raise ValueError("T5Config: d_model, num_heads, d_kv must be > 0")


class T5LayerNorm(nn.Module):
    """T5-style RMSNorm (no mean centering, no bias)."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.normalized_shape = (d_model,)

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.normalized_shape, self.weight, self.eps)


class T5DenseGatedActDense(nn.Module):
    """Gated FF: ``wo(act(wi_0(x)) * wi_1(x))`` (GELU tanh / gelu_new)."""

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: Tensor) -> Tensor:
        return self.wo(self.act(self.wi_0(x)) * self.wi_1(x))


class T5Attention(nn.Module):
    """Simplified multi-head attention (no relative bias buckets)."""

    def __init__(self, config: T5Config, *, is_decoder: bool = False) -> None:
        super().__init__()
        self.is_decoder = is_decoder
        self.n_heads = config.num_heads
        self.key_value_proj_dim = config.d_kv
        self.inner_dim = config.inner_dim
        self.d_model = config.d_model
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

    def _shape(self, x: Tensor) -> Tensor:
        b, s, _ = x.shape
        return x.view(
            b, s, self.n_heads, self.key_value_proj_dim
        ).transpose(1, 2)

    def forward(
        self,
        hidden: Tensor,
        key_value_states: Tensor | None = None,
        attn_mask: Tensor | None = None,
        *,
        is_causal: bool = False,
    ) -> Tensor:
        b, s, _ = hidden.shape
        q = self._shape(self.q(hidden))
        kv_input = hidden if key_value_states is None else key_value_states
        k = self._shape(self.k(kv_input))
        v = self._shape(self.v(kv_input))
        # HF T5 scores are unscaled matmuls; cancel NNTile SDPA's 1/sqrt(d).
        scale = float(self.key_value_proj_dim) ** 0.5
        if q.device.type == "nntile":
            q = q * torch.full(
                q.shape, scale, dtype=torch.float32, device="cpu"
            ).to(q.device)
        else:
            q = q * scale
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal and attn_mask is None,
        )
        out = out.transpose(1, 2).contiguous().view(b, s, self.inner_dim)
        return self.o(out)


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config) -> None:
        super().__init__()
        self.layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.DenseReluDense = T5DenseGatedActDense(config)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.DenseReluDense(self.layer_norm(x))


class T5EncoderBlock(nn.Module):
    def __init__(self, config: T5Config) -> None:
        super().__init__()
        self.layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.self_attn = T5Attention(config, is_decoder=False)
        self.ff = T5LayerFF(config)

    def forward(
        self, x: Tensor, attn_mask: Tensor | None = None
    ) -> Tensor:
        x = x + self.self_attn(self.layer_norm(x), attn_mask=attn_mask)
        return self.ff(x)


class T5DecoderBlock(nn.Module):
    def __init__(self, config: T5Config) -> None:
        super().__init__()
        eps = config.layer_norm_epsilon
        self.layer_norm_0 = T5LayerNorm(config.d_model, eps=eps)
        self.self_attn = T5Attention(config, is_decoder=True)
        self.layer_norm_1 = T5LayerNorm(config.d_model, eps=eps)
        self.cross_attn = T5Attention(config, is_decoder=True)
        self.ff = T5LayerFF(config)

    def forward(
        self,
        x: Tensor,
        encoder_hidden: Tensor,
        self_attn_mask: Tensor | None = None,
        cross_attn_mask: Tensor | None = None,
    ) -> Tensor:
        x = x + self.self_attn(
            self.layer_norm_0(x),
            attn_mask=self_attn_mask,
            is_causal=self_attn_mask is None,
        )
        x = x + self.cross_attn(
            self.layer_norm_1(x),
            key_value_states=encoder_hidden,
            attn_mask=cross_attn_mask,
            is_causal=False,
        )
        return self.ff(x)


class T5Stack(nn.Module):
    def __init__(
        self,
        config: T5Config,
        embed_tokens: nn.Embedding,
        *,
        is_decoder: bool,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.embed_tokens = embed_tokens
        self.is_decoder = is_decoder
        if is_decoder:
            self.block = nn.ModuleList(
                [T5DecoderBlock(config) for _ in range(num_layers)]
            )
        else:
            self.block = nn.ModuleList(
                [T5EncoderBlock(config) for _ in range(num_layers)]
            )
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )

    def forward(
        self,
        input_ids: Tensor,
        encoder_hidden: Tensor | None = None,
        attn_mask: Tensor | None = None,
        cross_attn_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.embed_tokens(input_ids)
        if self.is_decoder:
            assert encoder_hidden is not None
            for block in self.block:
                x = block(
                    x,
                    encoder_hidden,
                    self_attn_mask=attn_mask,
                    cross_attn_mask=cross_attn_mask,
                )
        else:
            for block in self.block:
                x = block(x, attn_mask)
        return self.final_layer_norm(x)


class T5Model(nn.Module):
    def __init__(self, config: T5Config) -> None:
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = T5Stack(
            config,
            self.shared,
            is_decoder=False,
            num_layers=config.num_layers,
        )
        self.decoder = T5Stack(
            config,
            self.shared,
            is_decoder=True,
            num_layers=config.num_decoder_layers,
        )

    def forward(
        self,
        encoder_input_ids: Tensor,
        decoder_input_ids: Tensor,
        encoder_attention_mask: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
        cross_attention_mask: Tensor | None = None,
    ) -> Tensor:
        enc = self.encoder(
            encoder_input_ids, attn_mask=encoder_attention_mask
        )
        return self.decoder(
            decoder_input_ids,
            encoder_hidden=enc,
            attn_mask=decoder_attention_mask,
            cross_attn_mask=cross_attention_mask,
        )


class T5ForConditionalGeneration(nn.Module):
    """Encoder-decoder + lm_head (simplified relative to full HF T5)."""

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.model = T5Model(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.shared.weight

    def forward(
        self,
        encoder_input_ids: Tensor,
        decoder_input_ids: Tensor,
        encoder_attention_mask: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
        cross_attention_mask: Tensor | None = None,
    ) -> Tensor:
        hidden = self.model(
            encoder_input_ids,
            decoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            cross_attention_mask=cross_attention_mask,
        )
        if self.config.tie_word_embeddings:
            # Prefer mul.Scalar path; avoid 0-d broadcast issues on nntile.
            scale = float(self.config.d_model ** -0.5)
            hidden = hidden * scale
        return self.lm_head(hidden)


__all__ = [
    "T5Attention",
    "T5Config",
    "T5DecoderBlock",
    "T5EncoderBlock",
    "T5ForConditionalGeneration",
    "T5LayerFF",
    "T5Model",
    "T5Stack",
]
