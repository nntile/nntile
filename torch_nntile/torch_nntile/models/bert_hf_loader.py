# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/bert_hf_loader.py
# Convert weights between HuggingFace BERT and torch_nntile BertMlm.

"""Bidirectional HF <-> NNTile weight conversion for BERT MLM."""

from __future__ import annotations

import torch
from transformers import BertConfig as HfBertConfig
from transformers import BertForMaskedLM

from torch_nntile.models.bert import BertConfig, BertMlm
from torch_nntile.models.hf_rope_layout import copy_linear
from torch_nntile.nn.linear import (
    linear_to_output_weight,
    linear_to_qkv_bias,
    linear_to_qkv_weight,
    output_to_linear_weight,
    qkv_to_linear_bias,
    qkv_to_linear_weight,
)


def bert_config_from_hf(hf: HfBertConfig) -> BertConfig:
    """Build a local ``BertConfig`` from an HF config."""
    return BertConfig(
        vocab_size=int(hf.vocab_size),
        hidden_size=int(hf.hidden_size),
        intermediate_size=int(hf.intermediate_size),
        num_hidden_layers=int(hf.num_hidden_layers),
        num_attention_heads=int(hf.num_attention_heads),
        max_position_embeddings=int(hf.max_position_embeddings),
        type_vocab_size=int(hf.type_vocab_size),
        layer_norm_eps=float(hf.layer_norm_eps),
        hidden_act=str(hf.hidden_act),
    )


def _load_embeddings(dst, src) -> None:
    dst.word_embeddings.weight.data.copy_(src.word_embeddings.weight.data)
    dst.position_embeddings.weight.data.copy_(
        src.position_embeddings.weight.data
    )
    dst.token_type_embeddings.weight.data.copy_(
        src.token_type_embeddings.weight.data
    )
    dst.LayerNorm.weight.data.copy_(src.LayerNorm.weight.data)
    dst.LayerNorm.bias.data.copy_(src.LayerNorm.bias.data)


def _load_layer(dst, src) -> None:
    self_attn = dst.attention.self
    n_heads = self_attn.n_heads
    head_size = self_attn.head_dim
    self_attn.query.weight.data.copy_(
        linear_to_qkv_weight(
            src.attention.self.query.weight.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )
    self_attn.key.weight.data.copy_(
        linear_to_qkv_weight(
            src.attention.self.key.weight.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )
    self_attn.value.weight.data.copy_(
        linear_to_qkv_weight(
            src.attention.self.value.weight.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )
    self_attn.query.bias.data.copy_(
        linear_to_qkv_bias(
            src.attention.self.query.bias.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )
    self_attn.key.bias.data.copy_(
        linear_to_qkv_bias(
            src.attention.self.key.bias.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )
    self_attn.value.bias.data.copy_(
        linear_to_qkv_bias(
            src.attention.self.value.bias.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )
    dst.attention.output.dense.weight.data.copy_(
        linear_to_output_weight(
            src.attention.output.dense.weight.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )
    dst.attention.output.dense.bias.data.copy_(
        src.attention.output.dense.bias.data
    )
    dst.attention.output.LayerNorm.weight.data.copy_(
        src.attention.output.LayerNorm.weight.data
    )
    dst.attention.output.LayerNorm.bias.data.copy_(
        src.attention.output.LayerNorm.bias.data
    )
    copy_linear(dst.intermediate.dense, src.intermediate.dense)
    copy_linear(dst.output.dense, src.output.dense)
    dst.output.LayerNorm.weight.data.copy_(src.output.LayerNorm.weight.data)
    dst.output.LayerNorm.bias.data.copy_(src.output.LayerNorm.bias.data)


def load_hf_into_bert_mlm(minimal: BertMlm, hf: BertForMaskedLM) -> None:
    """Copy HF weights into ``minimal`` (CPU; call ``.to('nntile')`` after)."""
    _load_embeddings(minimal.bert.embeddings, hf.bert.embeddings)
    for dst_layer, src_layer in zip(
        minimal.bert.encoder.layer, hf.bert.encoder.layer
    ):
        _load_layer(dst_layer, src_layer)

    pred = hf.cls.predictions
    transform = pred.transform
    minimal.cls.dense.weight.data.copy_(transform.dense.weight.data)
    minimal.cls.dense.bias.data.copy_(transform.dense.bias.data)
    minimal.cls.LayerNorm.weight.data.copy_(transform.LayerNorm.weight.data)
    minimal.cls.LayerNorm.bias.data.copy_(transform.LayerNorm.bias.data)
    # Untied locally: copy HF decoder (or tied embedding) values by value.
    minimal.cls.decoder.weight.data.copy_(pred.decoder.weight.data)
    if pred.decoder.bias is not None:
        minimal.cls.decoder.bias.data.copy_(pred.decoder.bias.data)
    elif hasattr(pred, "bias") and pred.bias is not None:
        minimal.cls.decoder.bias.data.copy_(pred.bias.data)


def export_bert_mlm_to_hf_state_dict(
    minimal: BertMlm,
    *,
    config: HfBertConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Export local BERT CPU weights as an HF state dict."""
    cfg = minimal.config
    if config is None:
        config = HfBertConfig(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_hidden_layers=cfg.num_hidden_layers,
            num_attention_heads=cfg.num_attention_heads,
            max_position_embeddings=cfg.max_position_embeddings,
            type_vocab_size=cfg.type_vocab_size,
            layer_norm_eps=cfg.layer_norm_eps,
            hidden_act=cfg.hidden_act,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
    config._attn_implementation = "eager"
    hf = BertForMaskedLM(config).float()
    _load_embeddings(hf.bert.embeddings, minimal.bert.embeddings)
    for dst_layer, src_layer in zip(
        hf.bert.encoder.layer, minimal.bert.encoder.layer
    ):
        src_attn = src_layer.attention.self
        n_heads = src_attn.n_heads
        head_size = src_attn.head_dim
        dst_layer.attention.self.query.weight.data.copy_(
            qkv_to_linear_weight(src_attn.query.weight.data)
        )
        dst_layer.attention.self.key.weight.data.copy_(
            qkv_to_linear_weight(src_attn.key.weight.data)
        )
        dst_layer.attention.self.value.weight.data.copy_(
            qkv_to_linear_weight(src_attn.value.weight.data)
        )
        del n_heads, head_size
        dst_layer.attention.self.query.bias.data.copy_(
            qkv_to_linear_bias(src_attn.query.bias.data)
        )
        dst_layer.attention.self.key.bias.data.copy_(
            qkv_to_linear_bias(src_attn.key.bias.data)
        )
        dst_layer.attention.self.value.bias.data.copy_(
            qkv_to_linear_bias(src_attn.value.bias.data)
        )
        dst_layer.attention.output.dense.weight.data.copy_(
            output_to_linear_weight(
                src_layer.attention.output.dense.weight.data
            )
        )
        dst_layer.attention.output.dense.bias.data.copy_(
            src_layer.attention.output.dense.bias.data
        )
        dst_layer.attention.output.LayerNorm.weight.data.copy_(
            src_layer.attention.output.LayerNorm.weight.data
        )
        dst_layer.attention.output.LayerNorm.bias.data.copy_(
            src_layer.attention.output.LayerNorm.bias.data
        )
        copy_linear(dst_layer.intermediate.dense, src_layer.intermediate.dense)
        copy_linear(dst_layer.output.dense, src_layer.output.dense)
        dst_layer.output.LayerNorm.weight.data.copy_(
            src_layer.output.LayerNorm.weight.data
        )
        dst_layer.output.LayerNorm.bias.data.copy_(
            src_layer.output.LayerNorm.bias.data
        )

    pred = hf.cls.predictions
    transform = pred.transform
    transform.dense.weight.data.copy_(minimal.cls.dense.weight.data)
    transform.dense.bias.data.copy_(minimal.cls.dense.bias.data)
    transform.LayerNorm.weight.data.copy_(minimal.cls.LayerNorm.weight.data)
    transform.LayerNorm.bias.data.copy_(minimal.cls.LayerNorm.bias.data)
    pred.decoder.weight.data.copy_(minimal.cls.decoder.weight.data)
    pred.decoder.bias.data.copy_(minimal.cls.decoder.bias.data)
    if hasattr(pred, "bias") and pred.bias is not None:
        pred.bias.data.copy_(minimal.cls.decoder.bias.data)
    # Keep decoder untied (migration debt: no shared Parameter storage).

    with torch.no_grad():
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in hf.state_dict().items()
        }


__all__ = [
    "bert_config_from_hf",
    "export_bert_mlm_to_hf_state_dict",
    "load_hf_into_bert_mlm",
]
