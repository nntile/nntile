# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_roberta_hf_parity.py
# RoBERTa layer + full-model parity vs HuggingFace RobertaForMaskedLM.

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("transformers")

import torch
from transformers import RobertaConfig as HfRobertaConfig
from transformers import RobertaForMaskedLM
from transformers.models.roberta.modeling_roberta import (
    RobertaEmbeddings as HfEmbeddings,
)

import torch_nntile
from torch_nntile import _C
from torch_nntile.models.bert import BertLayer
from torch_nntile.models.hf_rope_layout import copy_linear
from torch_nntile.models.roberta import RobertaEmbeddings, RobertaMlm
from torch_nntile.models.roberta_hf_loader import (
    load_hf_into_roberta_mlm,
    roberta_config_from_hf,
)
from parity_helpers import assert_close, contiguous_to_nntile


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)

RTOL = 1e-4
ATOL = 1e-4


@pytest.fixture
def tiny_hf_config() -> HfRobertaConfig:
    cfg = HfRobertaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=40,
        type_vocab_size=1,
        pad_token_id=1,
        layer_norm_eps=1e-5,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _make_models(hf_cfg: HfRobertaConfig):
    torch.manual_seed(0)
    hf = RobertaForMaskedLM(hf_cfg).eval().float()
    local_cfg = roberta_config_from_hf(hf_cfg)
    minimal = RobertaMlm(local_cfg).eval().float()
    load_hf_into_roberta_mlm(minimal, hf)
    minimal = minimal.to("nntile")
    return hf, minimal


def test_roberta_embeddings_forward_matches_hf(tiny_hf_config):
    torch.manual_seed(1)
    hf_emb = HfEmbeddings(tiny_hf_config).eval().float()
    local = RobertaEmbeddings(
        roberta_config_from_hf(tiny_hf_config)
    ).eval().float()
    local.word_embeddings.weight.data.copy_(hf_emb.word_embeddings.weight.data)
    local.position_embeddings.weight.data.copy_(
        hf_emb.position_embeddings.weight.data
    )
    local.LayerNorm.load_state_dict(hf_emb.LayerNorm.state_dict())
    local = local.to("nntile")
    # Include pad tokens so pad-aware position ids are exercised.
    input_ids = torch.randint(2, tiny_hf_config.vocab_size, (2, 8))
    input_ids[:, 0] = tiny_hf_config.pad_token_id
    with torch.no_grad():
        ref = hf_emb(input_ids)
        out = local(contiguous_to_nntile(input_ids))
    assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_roberta_layer_forward_matches_hf(tiny_hf_config):
    torch.manual_seed(2)
    hf = RobertaForMaskedLM(tiny_hf_config).eval().float()
    hf_layer = hf.roberta.encoder.layer[0]
    local_cfg = roberta_config_from_hf(tiny_hf_config)
    local = BertLayer(local_cfg.to_bert_config()).eval().float()
    copy_linear(local.attention.self.query, hf_layer.attention.self.query)
    copy_linear(local.attention.self.key, hf_layer.attention.self.key)
    copy_linear(local.attention.self.value, hf_layer.attention.self.value)
    copy_linear(local.attention.output.dense, hf_layer.attention.output.dense)
    local.attention.output.LayerNorm.load_state_dict(
        hf_layer.attention.output.LayerNorm.state_dict()
    )
    copy_linear(local.intermediate.dense, hf_layer.intermediate.dense)
    copy_linear(local.output.dense, hf_layer.output.dense)
    local.output.LayerNorm.load_state_dict(hf_layer.output.LayerNorm.state_dict())
    local = local.to("nntile")
    x = torch.randn(2, 8, tiny_hf_config.hidden_size)
    with torch.no_grad():
        ref = hf_layer(x)[0]
        out = local(contiguous_to_nntile(x), is_causal=False)
    assert_close(out, ref, rtol=RTOL, atol=5e-4)


def test_roberta_mlm_forward_matches_hf(tiny_hf_config):
    hf, minimal = _make_models(tiny_hf_config)
    input_ids = torch.randint(2, tiny_hf_config.vocab_size, (2, 8))
    input_ids[:, 0] = tiny_hf_config.pad_token_id
    with torch.no_grad():
        ref = hf(input_ids).logits
        out = minimal(contiguous_to_nntile(input_ids))
    assert_close(out, ref, rtol=RTOL, atol=5e-4)


def test_roberta_mlm_backward_matches_hf(tiny_hf_config):
    hf, minimal = _make_models(tiny_hf_config)
    for p in hf.parameters():
        p.requires_grad_(True)
    for p in minimal.parameters():
        p.requires_grad_(True)
    input_ids = torch.randint(2, tiny_hf_config.vocab_size, (2, 8))
    grad = torch.randn(2, 8, tiny_hf_config.vocab_size)
    logits_ref = hf(input_ids).logits
    logits_ref.backward(grad)
    logits = minimal(contiguous_to_nntile(input_ids))
    (gw,) = torch.autograd.grad(
        logits,
        minimal.roberta.embeddings.word_embeddings.weight,
        grad_outputs=contiguous_to_nntile(grad),
    )
    assert_close(
        gw,
        hf.roberta.embeddings.word_embeddings.weight.grad,
        rtol=1e-3,
        atol=1e-3,
    )
