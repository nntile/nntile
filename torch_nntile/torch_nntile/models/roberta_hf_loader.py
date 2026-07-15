# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/roberta_hf_loader.py
# Convert weights between HuggingFace RoBERTa and torch_nntile RobertaMlm.

"""Bidirectional HF ↔ NNTile weight conversion for RoBERTa MLM."""

from __future__ import annotations

import torch
from transformers import RobertaConfig as HfRobertaConfig
from transformers import RobertaForMaskedLM

from torch_nntile.models.bert_hf_loader import _load_layer
from torch_nntile.models.hf_rope_layout import copy_linear
from torch_nntile.models.roberta import RobertaConfig, RobertaMlm


def roberta_config_from_hf(hf: HfRobertaConfig) -> RobertaConfig:
    """Build a local ``RobertaConfig`` from an HF config."""
    return RobertaConfig(
        vocab_size=int(hf.vocab_size),
        hidden_size=int(hf.hidden_size),
        intermediate_size=int(hf.intermediate_size),
        num_hidden_layers=int(hf.num_hidden_layers),
        num_attention_heads=int(hf.num_attention_heads),
        max_position_embeddings=int(hf.max_position_embeddings),
        type_vocab_size=int(getattr(hf, "type_vocab_size", 1)),
        pad_token_id=int(hf.pad_token_id),
        layer_norm_eps=float(hf.layer_norm_eps),
        hidden_act=str(hf.hidden_act),
    )


def _load_embeddings(dst, src) -> None:
    dst.word_embeddings.weight.data.copy_(src.word_embeddings.weight.data)
    dst.position_embeddings.weight.data.copy_(
        src.position_embeddings.weight.data
    )
    dst.LayerNorm.weight.data.copy_(src.LayerNorm.weight.data)
    dst.LayerNorm.bias.data.copy_(src.LayerNorm.bias.data)


def load_hf_into_roberta_mlm(
    minimal: RobertaMlm,
    hf: RobertaForMaskedLM,
) -> None:
    """Copy HF weights into ``minimal`` (CPU; call ``.to('nntile')`` after)."""
    _load_embeddings(minimal.roberta.embeddings, hf.roberta.embeddings)
    for dst_layer, src_layer in zip(
        minimal.roberta.encoder.layer, hf.roberta.encoder.layer
    ):
        _load_layer(dst_layer, src_layer)

    head = hf.lm_head
    copy_linear(minimal.lm_head.dense, head.dense)
    minimal.lm_head.layer_norm.weight.data.copy_(head.layer_norm.weight.data)
    minimal.lm_head.layer_norm.bias.data.copy_(head.layer_norm.bias.data)
    minimal.lm_head.decoder.weight.data.copy_(head.decoder.weight.data)
    # HF keeps an extra ``lm_head.bias``; local bias lives on the decoder.
    if head.bias is not None:
        minimal.lm_head.decoder.bias.data.copy_(head.bias.data)
    elif head.decoder.bias is not None:
        minimal.lm_head.decoder.bias.data.copy_(head.decoder.bias.data)


def export_roberta_mlm_to_hf_state_dict(
    minimal: RobertaMlm,
    *,
    config: HfRobertaConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Export local RoBERTa CPU weights as an HF state_dict."""
    cfg = minimal.config
    if config is None:
        config = HfRobertaConfig(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_hidden_layers=cfg.num_hidden_layers,
            num_attention_heads=cfg.num_attention_heads,
            max_position_embeddings=cfg.max_position_embeddings,
            type_vocab_size=cfg.type_vocab_size,
            pad_token_id=cfg.pad_token_id,
            layer_norm_eps=cfg.layer_norm_eps,
            hidden_act=cfg.hidden_act,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
    config._attn_implementation = "eager"
    hf = RobertaForMaskedLM(config).float()
    _load_embeddings(hf.roberta.embeddings, minimal.roberta.embeddings)
    for dst_layer, src_layer in zip(
        hf.roberta.encoder.layer, minimal.roberta.encoder.layer
    ):
        _load_layer(dst_layer, src_layer)

    head = hf.lm_head
    copy_linear(head.dense, minimal.lm_head.dense)
    head.layer_norm.weight.data.copy_(minimal.lm_head.layer_norm.weight.data)
    head.layer_norm.bias.data.copy_(minimal.lm_head.layer_norm.bias.data)
    head.decoder.weight.data.copy_(minimal.lm_head.decoder.weight.data)
    head.bias.data.copy_(minimal.lm_head.decoder.bias.data)
    if head.decoder.bias is not None:
        head.decoder.bias.data.copy_(minimal.lm_head.decoder.bias.data)

    with torch.no_grad():
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in hf.state_dict().items()
        }


__all__ = [
    "export_roberta_mlm_to_hf_state_dict",
    "load_hf_into_roberta_mlm",
    "roberta_config_from_hf",
]
