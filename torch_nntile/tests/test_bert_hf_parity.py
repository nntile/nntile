# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_bert_hf_parity.py
# Thorough BERT submodule parity vs HuggingFace (mirrors deleted NNGraph
# bert_{config,intermediate,attention,layer,embeddings,model,mlm} matrix).

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("transformers")

import torch
from transformers import BertConfig as HfBertConfig
from transformers import BertForMaskedLM
from transformers.models.bert.modeling_bert import (
    BertAttention as HfAttention,
    BertEmbeddings as HfEmbeddings,
    BertIntermediate as HfIntermediate,
    BertLayer as HfLayer,
    BertSelfAttention as HfSelfAttention,
)

from torch_nntile import _C
from torch_nntile.models.bert import (
    BertAttention,
    BertConfig,
    BertEmbeddings,
    BertIntermediate,
    BertLayer,
    BertMlm,
    BertMlmHead,
    BertSelfAttention,
)
from torch_nntile.models.bert_hf_loader import (
    bert_config_from_hf,
    export_bert_mlm_to_hf_state_dict,
    load_hf_into_bert_mlm,
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
def tiny_hf_config() -> HfBertConfig:
    cfg = HfBertConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
        type_vocab_size=2,
        layer_norm_eps=1e-12,
        hidden_act="gelu_new",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _bert_input_ids(cfg: HfBertConfig) -> torch.Tensor:
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)
    input_ids[0, 0] = 0
    input_ids[1, -1] = cfg.vocab_size - 1
    return input_ids


def _bert_token_type_ids(cfg: HfBertConfig) -> torch.Tensor:
    return torch.randint(0, cfg.type_vocab_size, (2, 8), dtype=torch.long)


def _load_embeddings(local: BertEmbeddings, hf_emb: HfEmbeddings) -> None:
    local.word_embeddings.weight.data.copy_(hf_emb.word_embeddings.weight.data)
    local.position_embeddings.weight.data.copy_(
        hf_emb.position_embeddings.weight.data
    )
    local.token_type_embeddings.weight.data.copy_(
        hf_emb.token_type_embeddings.weight.data
    )
    local.LayerNorm.load_state_dict(hf_emb.LayerNorm.state_dict())


def _load_self_attention(
    local: BertSelfAttention, hf_self: HfSelfAttention
) -> None:
    copy_linear(local.query, hf_self.query)
    copy_linear(local.key, hf_self.key)
    copy_linear(local.value, hf_self.value)


def _load_attention(local: BertAttention, hf_attn: HfAttention) -> None:
    _load_self_attention(local.self, hf_attn.self)
    copy_linear(local.output.dense, hf_attn.output.dense)
    local.output.LayerNorm.load_state_dict(hf_attn.output.LayerNorm.state_dict())


def _load_layer(local: BertLayer, hf_layer: HfLayer) -> None:
    _load_attention(local.attention, hf_layer.attention)
    copy_linear(local.intermediate.dense, hf_layer.intermediate.dense)
    copy_linear(local.output.dense, hf_layer.output.dense)
    local.output.LayerNorm.load_state_dict(hf_layer.output.LayerNorm.state_dict())


def _make_models(hf_cfg: HfBertConfig) -> tuple[BertForMaskedLM, BertMlm]:
    torch.manual_seed(0)
    hf = BertForMaskedLM(hf_cfg).eval().float()
    local = BertMlm(bert_config_from_hf(hf_cfg)).eval().float()
    load_hf_into_bert_mlm(local, hf)
    return hf, local.to("nntile")


def _assert_forward_backward(
    *,
    x: torch.Tensor,
    ref_forward,
    local_forward,
    ref_weight: torch.Tensor,
    local_weight: torch.Tensor,
    atol: float = ATOL,
) -> None:
    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = ref_forward(x_ref)
    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)

    x_nnt = contiguous_to_nntile(x.detach()).requires_grad_(True)
    y = local_forward(x_nnt)
    assert_close(y, y_ref.detach(), rtol=RTOL, atol=atol)
    gx, gw = torch.autograd.grad(
        y,
        (x_nnt, local_weight),
        grad_outputs=contiguous_to_nntile(grad),
    )
    assert_close(gx, x_ref.grad, rtol=1e-3, atol=BWD_ATOL)
    assert_close(gw, ref_weight.grad, rtol=1e-3, atol=BWD_ATOL)


def test_bert_config_validate_head_dim_hidden_act(tiny_hf_config):
    cfg = bert_config_from_hf(tiny_hf_config)
    assert cfg.head_dim == 16
    assert cfg.hidden_act == "gelu_new"
    cfg.validate()

    with pytest.raises(ValueError):
        BertConfig(hidden_size=65, num_attention_heads=4).validate()
    with pytest.raises(ValueError):
        BertIntermediate(BertConfig(hidden_act="unsupported"))


@pytest.mark.parametrize("hidden_act", ["gelu", "gelu_new", "relu"])
def test_bert_intermediate_activation_variants(hidden_act, tiny_hf_config):
    tiny_hf_config.hidden_act = hidden_act
    torch.manual_seed(2)
    hf_inter = HfIntermediate(tiny_hf_config).eval().float()
    local = BertIntermediate(
        bert_config_from_hf(tiny_hf_config)
    ).eval().float()
    copy_linear(local.dense, hf_inter.dense)
    local = local.to("nntile")
    x = torch.randn(2, 8, tiny_hf_config.hidden_size)
    with torch.no_grad():
        assert_close(
            local(contiguous_to_nntile(x)),
            hf_inter(x),
            rtol=RTOL,
            atol=ATOL,
        )


def test_bert_embeddings_forward_matches_hf(tiny_hf_config):
    torch.manual_seed(1)
    hf_emb = HfEmbeddings(tiny_hf_config).eval().float()
    local = BertEmbeddings(bert_config_from_hf(tiny_hf_config)).eval().float()
    _load_embeddings(local, hf_emb)
    local = local.to("nntile")

    input_ids = _bert_input_ids(tiny_hf_config)
    token_type_ids = _bert_token_type_ids(tiny_hf_config)
    with torch.no_grad():
        ref = hf_emb(input_ids=input_ids, token_type_ids=token_type_ids)
        out = local(
            contiguous_to_nntile(input_ids),
            token_type_ids=contiguous_to_nntile(token_type_ids),
        )
    assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_bert_intermediate_forward_backward_matches_hf(tiny_hf_config):
    torch.manual_seed(2)
    hf_inter = HfIntermediate(tiny_hf_config).eval().float()
    local = BertIntermediate(
        bert_config_from_hf(tiny_hf_config)
    ).eval().float()
    copy_linear(local.dense, hf_inter.dense)
    local = local.to("nntile")

    x = torch.randn(2, 8, tiny_hf_config.hidden_size)
    _assert_forward_backward(
        x=x,
        ref_forward=hf_inter,
        local_forward=local,
        ref_weight=hf_inter.dense.weight,
        local_weight=local.dense.weight,
    )


def test_bert_self_attention_forward_backward_matches_hf(tiny_hf_config):
    torch.manual_seed(3)
    hf_self = HfSelfAttention(tiny_hf_config).eval().float()
    local = BertSelfAttention(
        bert_config_from_hf(tiny_hf_config)
    ).eval().float()
    _load_self_attention(local, hf_self)
    local = local.to("nntile")

    x = torch.randn(2, 8, tiny_hf_config.hidden_size)

    def _hf_self(t):
        # HF SelfAttention returns (context, probs).
        return hf_self(t)[0]

    _assert_forward_backward(
        x=x,
        ref_forward=_hf_self,
        local_forward=local,
        ref_weight=hf_self.query.weight,
        local_weight=local.query.weight,
        atol=ATTN_ATOL,
    )


def test_bert_attention_forward_backward_matches_hf(tiny_hf_config):
    torch.manual_seed(4)
    hf_attn = HfAttention(tiny_hf_config).eval().float()
    local = BertAttention(bert_config_from_hf(tiny_hf_config)).eval().float()
    _load_attention(local, hf_attn)
    local = local.to("nntile")

    x = torch.randn(2, 8, tiny_hf_config.hidden_size)
    _assert_forward_backward(
        x=x,
        ref_forward=lambda t: hf_attn(t)[0],
        local_forward=local,
        ref_weight=hf_attn.self.query.weight,
        local_weight=local.self.query.weight,
        atol=ATTN_ATOL,
    )


def test_bert_layer_forward_backward_matches_hf(tiny_hf_config):
    torch.manual_seed(5)
    hf_layer = HfLayer(tiny_hf_config).eval().float()
    local = BertLayer(bert_config_from_hf(tiny_hf_config)).eval().float()
    _load_layer(local, hf_layer)
    local = local.to("nntile")

    x = torch.randn(2, 8, tiny_hf_config.hidden_size)
    _assert_forward_backward(
        x=x,
        ref_forward=lambda t: hf_layer(t)[0],
        local_forward=local,
        ref_weight=hf_layer.attention.self.query.weight,
        local_weight=local.attention.self.query.weight,
        atol=ATTN_ATOL,
    )


def test_bert_model_hidden_forward_backward_matches_hf(tiny_hf_config):
    hf, local = _make_models(tiny_hf_config)
    input_ids = _bert_input_ids(tiny_hf_config)
    token_type_ids = _bert_token_type_ids(tiny_hf_config)
    with torch.no_grad():
        ref = hf.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        ).last_hidden_state
        out = local.bert(
            contiguous_to_nntile(input_ids),
            token_type_ids=contiguous_to_nntile(token_type_ids),
        )
    assert_close(out, ref, rtol=RTOL, atol=ATTN_ATOL)

    for p in hf.parameters():
        p.requires_grad_(True)
    for p in local.parameters():
        p.requires_grad_(True)
    grad = torch.randn_like(ref)
    y_ref = hf.bert(
        input_ids=input_ids, token_type_ids=token_type_ids
    ).last_hidden_state
    y_ref.backward(grad)
    y = local.bert(
        contiguous_to_nntile(input_ids),
        token_type_ids=contiguous_to_nntile(token_type_ids),
    )
    (gw,) = torch.autograd.grad(
        y,
        local.bert.encoder.layer[0].attention.self.query.weight,
        contiguous_to_nntile(grad),
    )
    assert_close(
        gw,
        hf.bert.encoder.layer[0].attention.self.query.weight.grad,
        rtol=1e-3,
        atol=BWD_ATOL,
    )


def test_bert_mlm_head_forward_backward_matches_hf(tiny_hf_config):
    torch.manual_seed(6)
    hf, local_full = _make_models(tiny_hf_config)
    # Rebuild head-only local module with tied embeddings from HF.
    cfg = bert_config_from_hf(tiny_hf_config)
    emb = BertEmbeddings(cfg).eval().float()
    emb.word_embeddings.weight.data.copy_(
        hf.bert.embeddings.word_embeddings.weight.data
    )
    head = BertMlmHead(cfg, emb.word_embeddings).eval().float()
    pred = hf.cls.predictions
    head.dense.weight.data.copy_(pred.transform.dense.weight.data)
    head.dense.bias.data.copy_(pred.transform.dense.bias.data)
    head.LayerNorm.load_state_dict(pred.transform.LayerNorm.state_dict())
    if pred.decoder.bias is not None:
        head.decoder.bias.data.copy_(pred.decoder.bias.data)
    head = head.to("nntile")

    x = torch.randn(2, 8, tiny_hf_config.hidden_size, requires_grad=True)
    y_ref = pred(x)
    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)
    x_n = contiguous_to_nntile(x.detach()).requires_grad_(True)
    y = head(x_n)
    assert_close(y, y_ref.detach(), rtol=RTOL, atol=ATTN_ATOL)
    (gx,) = torch.autograd.grad(y, x_n, contiguous_to_nntile(grad))
    assert_close(gx, x.grad, rtol=1e-3, atol=BWD_ATOL)


def test_bert_mlm_loader_forward_matches_hf(tiny_hf_config):
    hf, local = _make_models(tiny_hf_config)
    input_ids = _bert_input_ids(tiny_hf_config)
    token_type_ids = _bert_token_type_ids(tiny_hf_config)
    with torch.no_grad():
        ref = hf(input_ids=input_ids, token_type_ids=token_type_ids).logits
        out = local(
            contiguous_to_nntile(input_ids),
            token_type_ids=contiguous_to_nntile(token_type_ids),
        )
    assert_close(out, ref, rtol=RTOL, atol=ATTN_ATOL)


def test_bert_mlm_logits_forward_backward_query_weight_matches_hf(
    tiny_hf_config,
):
    hf, local = _make_models(tiny_hf_config)
    input_ids = _bert_input_ids(tiny_hf_config)
    token_type_ids = _bert_token_type_ids(tiny_hf_config)
    grad = torch.randn(2, 8, tiny_hf_config.vocab_size)

    logits_ref = hf(input_ids=input_ids, token_type_ids=token_type_ids).logits
    logits_ref.backward(grad)
    logits = local(
        contiguous_to_nntile(input_ids),
        token_type_ids=contiguous_to_nntile(token_type_ids),
    )
    assert_close(logits, logits_ref.detach(), rtol=RTOL, atol=ATTN_ATOL)

    (gw,) = torch.autograd.grad(
        logits,
        local.bert.encoder.layer[0].attention.self.query.weight,
        grad_outputs=contiguous_to_nntile(grad),
    )
    assert_close(
        gw,
        hf.bert.encoder.layer[0].attention.self.query.weight.grad,
        rtol=1e-3,
        atol=BWD_ATOL,
    )


def test_bert_export_roundtrip_state_dict_matches_hf_keys(tiny_hf_config):
    torch.manual_seed(7)
    hf = BertForMaskedLM(tiny_hf_config).eval().float()
    local = BertMlm(bert_config_from_hf(tiny_hf_config)).eval().float()
    load_hf_into_bert_mlm(local, hf)
    exported = export_bert_mlm_to_hf_state_dict(local, config=tiny_hf_config)
    # Spot-check overlapping trainable keys.
    for key in (
        "bert.embeddings.word_embeddings.weight",
        "bert.encoder.layer.0.attention.self.query.weight",
        "cls.predictions.transform.dense.weight",
    ):
        assert key in exported
        torch.testing.assert_close(
            exported[key], hf.state_dict()[key], rtol=0, atol=0
        )
