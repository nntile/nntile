# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/roberta.py
# RoBERTa masked LM for device="nntile".

"""RoBERTa stack mirroring ``nntile::model::roberta`` (RobertaMlm)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from torch_nntile.models.bert import (
    BertConfig,
    BertEncoder,
    _bert_activation,
)


@dataclass
class RobertaConfig:
    vocab_size: int = 50265
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 514
    type_vocab_size: int = 1
    pad_token_id: int = 1
    layer_norm_eps: float = 1e-5
    hidden_act: str = "gelu"
    name: str = "roberta"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def validate(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "RobertaConfig: hidden_size must be divisible by "
                "num_attention_heads"
            )

    def to_bert_config(self) -> BertConfig:
        return BertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            layer_norm_eps=self.layer_norm_eps,
            hidden_act=self.hidden_act,
            name="bert",
        )


class RobertaEmbeddings(nn.Module):
    """Word + position embeddings (positions start at ``pad_token_id + 1``)."""

    def __init__(self, config: RobertaConfig) -> None:
        super().__init__()
        self.pad_token_id = config.pad_token_id
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=(
                config.pad_token_id if config.pad_token_id >= 0 else None
            ),
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        b, s = input_ids.shape
        if position_ids is None:
            # HF RoBERTa: positions are pad_token_id + 1 .. for non-pad tokens.
            # Compare on CPU — nntile lacks aten::ne for integer tensors.
            ids_cpu = input_ids.detach().to("cpu")
            pad = self.pad_token_id if self.pad_token_id >= 0 else -1
            if pad >= 0:
                mask = (ids_cpu != pad).to(torch.long)
                incremental = mask.cumsum(dim=1) * mask
                position_ids = incremental + pad
            else:
                position_ids = (
                    torch.arange(s, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(b, s)
                    .contiguous()
                )
            if input_ids.device.type != "cpu":
                position_ids = position_ids.contiguous().to(input_ids.device)
        x = self.word_embeddings(input_ids) + self.position_embeddings(
            position_ids
        )
        return self.LayerNorm(x)


class RobertaModel(nn.Module):
    def __init__(self, config: RobertaConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = BertEncoder(config.to_bert_config())

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        attn_mask: Tensor | None = None,
        *,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.embeddings(input_ids, position_ids)
        return self.encoder(x, attn_mask, is_causal=is_causal)


class RobertaMlmHead(nn.Module):
    def __init__(self, config: RobertaConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=True
        )
        # Match HF ``RobertaLMHead`` (ACT2FN[hidden_act]).
        self.act = _bert_activation(config.hidden_act)

    def forward(self, hidden: Tensor) -> Tensor:
        x = self.layer_norm(self.act(self.dense(hidden)))
        return self.decoder(x)


class RobertaMlm(nn.Module):
    """RoBERTa masked LM (``nntile::model::roberta::RobertaMlm``)."""

    def __init__(self, config: RobertaConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaMlmHead(config)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        attn_mask: Tensor | None = None,
        *,
        is_causal: bool = False,
    ) -> Tensor:
        hidden = self.roberta(
            input_ids,
            position_ids=position_ids,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )
        return self.lm_head(hidden)


__all__ = [
    "RobertaConfig",
    "RobertaEmbeddings",
    "RobertaMlm",
    "RobertaMlmHead",
    "RobertaModel",
]
