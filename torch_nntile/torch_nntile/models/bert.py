# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/bert.py
# BERT masked LM for device="nntile".

"""BERT stack mirroring ``nntile::model::bert`` (BertMlm)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from torch_nntile.nn.linear import (
    NntileAttentionOutput,
    NntileLinear,
    NntileQKVProjection,
    prepare_sdpa_mask,
)
from torch_nntile.nn.sdpa import sdpa_kernel


@dataclass
class BertConfig:
    vocab_size: int = 30522
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    hidden_act: str = "gelu"
    name: str = "bert"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def validate(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "BertConfig: hidden_size must be divisible by "
                "num_attention_heads"
            )


def _bert_activation(name: str) -> nn.Module:
    if name in ("gelu",):
        return nn.GELU()
    if name in ("gelu_pytorch_tanh", "gelutanh", "gelu_new"):
        return nn.GELU(approximate="tanh")
    if name == "relu":
        return nn.ReLU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    raise ValueError(f"BertConfig: unsupported hidden_act '{name}'")


class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self._position_ids_cache: dict[tuple[int, int], Tensor] = {}
        self._token_type_ids_cache: dict[tuple[int, int], Tensor] = {}

    def _cached_arange_ids(self, input_ids: Tensor) -> Tensor:
        batch, seq = int(input_ids.size(0)), int(input_ids.size(-1))
        key = (batch, seq)
        cached = self._position_ids_cache.get(key)
        if cached is not None and cached.device == input_ids.device:
            return cached
        ids = (
            torch.arange(seq, dtype=torch.long, device="cpu")
            .unsqueeze(0)
            .expand(batch, seq)
            .contiguous()
        )
        if input_ids.device.type != "cpu":
            ids = ids.to(input_ids.device)
        self._position_ids_cache[key] = ids
        return ids

    def _cached_zero_token_types(self, input_ids: Tensor) -> Tensor:
        batch, seq = int(input_ids.size(0)), int(input_ids.size(-1))
        key = (batch, seq)
        cached = self._token_type_ids_cache.get(key)
        if cached is not None and cached.device == input_ids.device:
            return cached
        ids = torch.zeros(batch, seq, dtype=torch.long, device="cpu")
        if input_ids.device.type != "cpu":
            ids = ids.to(input_ids.device)
        self._token_type_ids_cache[key] = ids
        return ids

    def clear_sequence_caches(self) -> None:
        self._position_ids_cache.clear()
        self._token_type_ids_cache.clear()

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        if position_ids is None:
            position_ids = self._cached_arange_ids(input_ids)
        if token_type_ids is None:
            token_type_ids = self._cached_zero_token_types(input_ids)
        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.LayerNorm(x)


class BertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden = config.hidden_size
        self.query = NntileQKVProjection(
            self.hidden,
            self.head_dim,
            self.n_heads,
            bias=True,
        )
        self.key = NntileQKVProjection(
            self.hidden,
            self.head_dim,
            self.n_heads,
            bias=True,
        )
        self.value = NntileQKVProjection(
            self.hidden,
            self.head_dim,
            self.n_heads,
            bias=True,
        )

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        *,
        is_causal: bool = False,
    ) -> Tensor:
        s = int(x.size(1))
        mask = prepare_sdpa_mask(
            attn_mask,
            x,
            q_len=s,
            is_causal=is_causal,
        )
        return sdpa_kernel(
            self.query(x),
            self.key(x),
            self.value(x),
            mask=mask,
            batch_ndim=2,
        )


class BertSelfOutput(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = NntileAttentionOutput(
            config.head_dim,
            config.num_attention_heads,
            config.hidden_size,
            bias=True,
        )
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, hidden: Tensor, input_tensor: Tensor) -> Tensor:
        return self.LayerNorm(self.dense(hidden) + input_tensor)


class BertAttention(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        *,
        is_causal: bool = False,
    ) -> Tensor:
        return self.output(self.self(x, attn_mask, is_causal=is_causal), x)


class BertIntermediate(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = NntileLinear(
            config.hidden_size,
            config.intermediate_size,
        )
        self.intermediate_act_fn = _bert_activation(config.hidden_act)

    def forward(self, x: Tensor) -> Tensor:
        return self.intermediate_act_fn(self.dense(x))


class BertOutput(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = NntileLinear(
            config.intermediate_size,
            config.hidden_size,
        )
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, hidden: Tensor, input_tensor: Tensor) -> Tensor:
        return self.LayerNorm(self.dense(hidden) + input_tensor)


class BertLayer(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        *,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.attention(x, attn_mask, is_causal=is_causal)
        return self.output(self.intermediate(x), x)


class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        *,
        is_causal: bool = False,
    ) -> Tensor:
        for layer in self.layer:
            x = layer(x, attn_mask, is_causal=is_causal)
        return x


class BertModel(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        attn_mask: Tensor | None = None,
        *,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.embeddings(input_ids, token_type_ids, position_ids)
        return self.encoder(x, attn_mask, is_causal=is_causal)


class BertMlmHead(nn.Module):
    """Prediction transform + vocab decoder (untied; see migration debt)."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = NntileLinear(config.hidden_size, config.hidden_size)
        # Match HF ``BertPredictionHeadTransform`` (ACT2FN[hidden_act]).
        self.transform_act_fn = _bert_activation(config.hidden_act)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.decoder = NntileLinear(
            config.hidden_size,
            config.vocab_size,
            bias=True,
        )

    def forward(self, hidden: Tensor) -> Tensor:
        x = self.LayerNorm(self.transform_act_fn(self.dense(hidden)))
        return self.decoder(x)


class BertMlm(nn.Module):
    """BERT masked LM (``nntile::model::bert::BertMlm``)."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.bert = BertModel(config)
        self.cls = BertMlmHead(config)

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        attn_mask: Tensor | None = None,
        *,
        is_causal: bool = False,
    ) -> Tensor:
        hidden = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )
        return self.cls(hidden)


__all__ = [
    "BertAttention",
    "BertConfig",
    "BertEmbeddings",
    "BertEncoder",
    "BertLayer",
    "BertMlm",
    "BertMlmHead",
    "BertModel",
]
