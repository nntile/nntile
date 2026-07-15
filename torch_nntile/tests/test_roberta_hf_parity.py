# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_roberta_hf_parity.py
# Thorough RoBERTa submodule parity vs HuggingFace (mirrors deleted NNGraph
# roberta_{config,intermediate,attention,layer,embeddings,model,mlm} matrix).

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("transformers")

import torch
from transformers import RobertaConfig as HfRobertaConfig
from transformers import RobertaForMaskedLM
from transformers.models.roberta.modeling_roberta import (
    RobertaAttention as HfAttention,
    RobertaEmbeddings as HfEmbeddings,
    RobertaIntermediate as HfIntermediate,
    RobertaLayer as HfLayer,
)

from torch_nntile import _C
from torch_nntile.models.bert import (
    BertAttention,
    BertIntermediate,
    BertLayer,
)
from torch_nntile.models.roberta import (
    RobertaConfig,
    RobertaEmbeddings,
    RobertaMlm,
)
from torch_nntile.models.roberta_hf_loader import (
    load_hf_into_roberta_mlm,
    roberta_config_from_hf,
)
from parity_helpers import assert_close, contiguous_to_nntile, copy_linear


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)

RTOL = 1e-4
ATOL = 1e-4
ATTN_ATOL = 5e-4
BWD_ATOL = 1e-3


@pytest.fixture
def tiny_hf_config() -> HfRobertaConfig:
    cfg = HfRobertaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=40,
        type_vocab_size=2,
        pad_token_id=1,
        layer_norm_eps=1e-5,
        hidden_act="gelu_new",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _roberta_input_ids(cfg: HfRobertaConfig) -> torch.Tensor:
    input_ids = torch.randint(4, cfg.vocab_size, (2, 8), dtype=torch.long)
    input_ids[0, 0] = cfg.pad_token_id
    input_ids[0, 3] = cfg.pad_token_id
    input_ids[1, -1] = cfg.pad_token_id
    return input_ids


def _roberta_token_type_ids(cfg: HfRobertaConfig) -> torch.Tensor:
    return torch.randint(0, cfg.type_vocab_size, (2, 8), dtype=torch.long)


def _load_embeddings(local: RobertaEmbeddings, hf_emb: HfEmbeddings) -> None:
    local.word_embeddings.weight.data.copy_(hf_emb.word_embeddings.weight.data)
    local.position_embeddings.weight.data.copy_(
        hf_emb.position_embeddings.weight.data
    )
    local.token_type_embeddings.weight.data.copy_(
        hf_emb.token_type_embeddings.weight.data
    )
    local.LayerNorm.load_state_dict(hf_emb.LayerNorm.state_dict())


def _load_attention(local: BertAttention, hf_attn: HfAttention) -> None:
    copy_linear(local.self.query, hf_attn.self.query)
    copy_linear(local.self.key, hf_attn.self.key)
    copy_linear(local.self.value, hf_attn.self.value)
    copy_linear(local.output.dense, hf_attn.output.dense)
    local.output.LayerNorm.load_state_dict(hf_attn.output.LayerNorm.state_dict())


def _load_layer(local: BertLayer, hf_layer: HfLayer) -> None:
    _load_attention(local.attention, hf_layer.attention)
    copy_linear(local.intermediate.dense, hf_layer.intermediate.dense)
    copy_linear(local.output.dense, hf_layer.output.dense)
    local.output.LayerNorm.load_state_dict(hf_layer.output.LayerNorm.state_dict())


def _make_models(
    hf_cfg: HfRobertaConfig,
) -> tuple[RobertaForMaskedLM, RobertaMlm]:
    torch.manual_seed(0)
    hf = RobertaForMaskedLM(hf_cfg).eval().float()
    local = RobertaMlm(roberta_config_from_hf(hf_cfg)).eval().float()
    load_hf_into_roberta_mlm(local, hf)
    return hf, local.to("nntile")


def test_roberta_config_validate_to_bert_config(tiny_hf_config):
    cfg = roberta_config_from_hf(tiny_hf_config)
    assert cfg.head_dim == 16
    assert cfg.hidden_act == "gelu_new"
    assert cfg.pad_token_id == tiny_hf_config.pad_token_id
    assert cfg.type_vocab_size == tiny_hf_config.type_vocab_size
    cfg.validate()

    bert_cfg = cfg.to_bert_config()
    assert bert_cfg.hidden_size == cfg.hidden_size
    assert bert_cfg.num_attention_heads == cfg.num_attention_heads
    assert bert_cfg.type_vocab_size == cfg.type_vocab_size
    assert bert_cfg.layer_norm_eps == cfg.layer_norm_eps
    assert bert_cfg.hidden_act == cfg.hidden_act

    with pytest.raises(ValueError):
        RobertaConfig(hidden_size=65, num_attention_heads=4).validate()


def test_roberta_embeddings_forward_with_pads_matches_hf(tiny_hf_config):
    torch.manual_seed(1)
    hf_emb = HfEmbeddings(tiny_hf_config).eval().float()
    local = RobertaEmbeddings(
        roberta_config_from_hf(tiny_hf_config)
    ).eval().float()
    _load_embeddings(local, hf_emb)

    assert local.word_embeddings.padding_idx is None
    assert local.position_embeddings.padding_idx is None
    assert local.token_type_embeddings.padding_idx is None
    local = local.to("nntile")

    input_ids = _roberta_input_ids(tiny_hf_config)
    token_type_ids = _roberta_token_type_ids(tiny_hf_config)
    with torch.no_grad():
        ref = hf_emb(input_ids=input_ids, token_type_ids=token_type_ids)
        out = local(
            contiguous_to_nntile(input_ids),
            token_type_ids=contiguous_to_nntile(token_type_ids),
        )
    assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_roberta_intermediate_forward_matches_hf(tiny_hf_config):
    torch.manual_seed(2)
    hf_inter = HfIntermediate(tiny_hf_config).eval().float()
    local_cfg = roberta_config_from_hf(tiny_hf_config).to_bert_config()
    local = BertIntermediate(local_cfg).eval().float()
    copy_linear(local.dense, hf_inter.dense)
    local = local.to("nntile")

    x = torch.randn(2, 8, tiny_hf_config.hidden_size)
    with torch.no_grad():
        ref = hf_inter(x)
        out = local(contiguous_to_nntile(x))
    assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_roberta_attention_forward_matches_hf(tiny_hf_config):
    torch.manual_seed(3)
    hf_attn = HfAttention(tiny_hf_config).eval().float()
    local_cfg = roberta_config_from_hf(tiny_hf_config).to_bert_config()
    local = BertAttention(local_cfg).eval().float()
    _load_attention(local, hf_attn)
    local = local.to("nntile")

    x = torch.randn(2, 8, tiny_hf_config.hidden_size)
    with torch.no_grad():
        ref = hf_attn(x)[0]
        out = local(contiguous_to_nntile(x))
    assert_close(out, ref, rtol=RTOL, atol=ATTN_ATOL)


def test_roberta_layer_forward_matches_hf(tiny_hf_config):
    torch.manual_seed(4)
    hf_layer = HfLayer(tiny_hf_config).eval().float()
    local_cfg = roberta_config_from_hf(tiny_hf_config).to_bert_config()
    local = BertLayer(local_cfg).eval().float()
    _load_layer(local, hf_layer)
    local = local.to("nntile")

    x = torch.randn(2, 8, tiny_hf_config.hidden_size)
    with torch.no_grad():
        ref = hf_layer(x)[0]
        out = local(contiguous_to_nntile(x))
    assert_close(out, ref, rtol=RTOL, atol=ATTN_ATOL)


def test_roberta_model_hidden_forward_matches_hf(tiny_hf_config):
    hf, local = _make_models(tiny_hf_config)
    input_ids = _roberta_input_ids(tiny_hf_config)
    with torch.no_grad():
        ref = hf.roberta(input_ids=input_ids).last_hidden_state
        out = local.roberta(contiguous_to_nntile(input_ids))
    assert_close(out, ref, rtol=RTOL, atol=ATTN_ATOL)


def test_roberta_mlm_logits_forward_backward_query_weight_matches_hf(
    tiny_hf_config,
):
    hf, local = _make_models(tiny_hf_config)
    input_ids = _roberta_input_ids(tiny_hf_config)
    grad = torch.randn(2, 8, tiny_hf_config.vocab_size)

    logits_ref = hf(input_ids=input_ids).logits
    logits_ref.backward(grad)
    logits = local(contiguous_to_nntile(input_ids))
    assert_close(logits, logits_ref.detach(), rtol=RTOL, atol=ATTN_ATOL)

    (gw,) = torch.autograd.grad(
        logits,
        local.roberta.encoder.layer[0].attention.self.query.weight,
        grad_outputs=contiguous_to_nntile(grad),
    )
    assert_close(
        gw,
        hf.roberta.encoder.layer[0].attention.self.query.weight.grad,
        rtol=1e-3,
        atol=BWD_ATOL,
    )
