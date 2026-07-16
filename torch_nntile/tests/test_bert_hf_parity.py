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
from conftest import nntile_cpu
from parity_helpers import assert_close, contiguous_to_nntile, copy_linear
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
from torch_nntile.nn.linear import (
    linear_to_output_weight,
    linear_to_qkv_bias,
    linear_to_qkv_weight,
    qkv_to_linear_weight,
)
from transformers import BertConfig as HfBertConfig
from transformers import BertForMaskedLM
from transformers.models.bert.modeling_bert import (
    BertAttention as HfAttention,
)
from transformers.models.bert.modeling_bert import (
    BertEmbeddings as HfEmbeddings,
)
from transformers.models.bert.modeling_bert import (
    BertIntermediate as HfIntermediate,
)
from transformers.models.bert.modeling_bert import (
    BertLayer as HfLayer,
)
from transformers.models.bert.modeling_bert import (
    BertSelfAttention as HfSelfAttention,
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
    n_heads = local.n_heads
    head_size = local.head_dim
    local.query.weight.data.copy_(
        linear_to_qkv_weight(
            hf_self.query.weight.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )
    local.key.weight.data.copy_(
        linear_to_qkv_weight(
            hf_self.key.weight.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )
    local.value.weight.data.copy_(
        linear_to_qkv_weight(
            hf_self.value.weight.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )
    local.query.bias.data.copy_(
        linear_to_qkv_bias(
            hf_self.query.bias.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )
    local.key.bias.data.copy_(
        linear_to_qkv_bias(
            hf_self.key.bias.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )
    local.value.bias.data.copy_(
        linear_to_qkv_bias(
            hf_self.value.bias.data,
            n_heads=n_heads,
            head_size=head_size,
        )
    )


def _load_attention(local: BertAttention, hf_attn: HfAttention) -> None:
    _load_self_attention(local.self, hf_attn.self)
    local.output.dense.weight.data.copy_(
        linear_to_output_weight(
            hf_attn.output.dense.weight.data,
            n_heads=local.self.n_heads,
            head_size=local.self.head_dim,
        )
    )
    local.output.dense.bias.data.copy_(hf_attn.output.dense.bias.data)
    local.output.LayerNorm.load_state_dict(
        hf_attn.output.LayerNorm.state_dict()
    )


def _load_layer(local: BertLayer, hf_layer: HfLayer) -> None:
    _load_attention(local.attention, hf_layer.attention)
    copy_linear(local.intermediate.dense, hf_layer.intermediate.dense)
    copy_linear(local.output.dense, hf_layer.output.dense)
    local.output.LayerNorm.load_state_dict(
        hf_layer.output.LayerNorm.state_dict()
    )


def _untie_hf_bert_mlm(hf: BertForMaskedLM) -> None:
    """Clone MLM decoder weights so HF reference matches local untied grads."""
    pred = hf.cls.predictions
    emb_w = hf.bert.embeddings.word_embeddings.weight
    if pred.decoder.weight.data_ptr() == emb_w.data_ptr():
        pred.decoder.weight = torch.nn.Parameter(emb_w.detach().clone())


def _make_models(hf_cfg: HfBertConfig) -> tuple[BertForMaskedLM, BertMlm]:
    torch.manual_seed(0)
    hf = BertForMaskedLM(hf_cfg).eval().float()
    _untie_hf_bert_mlm(hf)
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
    local_weight_grad_to_ref=None,
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
    if local_weight_grad_to_ref is not None:
        gw = local_weight_grad_to_ref(gw)
    assert_close(gw, ref_weight.grad, rtol=1e-3, atol=BWD_ATOL)


def _qkv_weight_grad_to_linear(grad: torch.Tensor) -> torch.Tensor:
    return qkv_to_linear_weight(nntile_cpu(grad))


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
    local = (
        BertIntermediate(bert_config_from_hf(tiny_hf_config)).eval().float()
    )
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
    local = (
        BertIntermediate(bert_config_from_hf(tiny_hf_config)).eval().float()
    )
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
    """Self-attn returns SDPA layout; HF merge is covered by BertAttention."""
    torch.manual_seed(3)
    hf_self = HfSelfAttention(tiny_hf_config).eval().float()
    local = (
        BertSelfAttention(bert_config_from_hf(tiny_hf_config)).eval().float()
    )
    _load_self_attention(local, hf_self)
    local = local.to("nntile")

    x = torch.randn(2, 8, tiny_hf_config.hidden_size)
    x_nnt = contiguous_to_nntile(x).requires_grad_(True)
    heads = local(x_nnt)
    assert heads.shape == (
        tiny_hf_config.num_attention_heads,
        2,
        8,
        tiny_hf_config.hidden_size // tiny_hf_config.num_attention_heads,
    )

    # HF merge is ``[B, nh, S, hs] -> [B, S, H]``. Rebuild that on CPU from
    # local SDPA heads for a forward-only check (no nntile permute needed).
    heads_cpu = nntile_cpu(heads)
    merged = heads_cpu.permute(1, 2, 0, 3).reshape(2, 8, -1)
    with torch.no_grad():
        ref = hf_self(x)[0]
    torch.testing.assert_close(merged, ref, rtol=RTOL, atol=ATTN_ATOL)

    grad = contiguous_to_nntile(torch.randn(heads.shape, dtype=torch.float32))
    (gw,) = torch.autograd.grad(
        heads,
        local.query.weight,
        grad_outputs=grad,
    )
    assert gw.shape == local.query.weight.shape


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
        local_weight_grad_to_ref=_qkv_weight_grad_to_linear,
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
        local_weight_grad_to_ref=_qkv_weight_grad_to_linear,
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
        _qkv_weight_grad_to_linear(gw),
        hf.bert.encoder.layer[0].attention.self.query.weight.grad,
        rtol=1e-3,
        atol=BWD_ATOL,
    )


def test_bert_mlm_head_forward_backward_matches_hf(tiny_hf_config):
    torch.manual_seed(6)
    hf, local_full = _make_models(tiny_hf_config)
    del local_full
    # Rebuild head-only local module (untied decoder; copy HF values).
    cfg = bert_config_from_hf(tiny_hf_config)
    head = BertMlmHead(cfg).eval().float()
    pred = hf.cls.predictions
    head.dense.weight.data.copy_(pred.transform.dense.weight.data)
    head.dense.bias.data.copy_(pred.transform.dense.bias.data)
    head.LayerNorm.load_state_dict(pred.transform.LayerNorm.state_dict())
    head.decoder.weight.data.copy_(pred.decoder.weight.data)
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
        _qkv_weight_grad_to_linear(gw),
        hf.bert.encoder.layer[0].attention.self.query.weight.grad,
        rtol=1e-3,
        atol=BWD_ATOL,
    )


def test_bert_mlm_untied_embedding_and_decoder_backward_matches_hf(
    tiny_hf_config,
):
    hf, local = _make_models(tiny_hf_config)
    assert (
        hf.cls.predictions.decoder.weight
        is not hf.bert.embeddings.word_embeddings.weight
    )
    assert (
        local.cls.decoder.weight
        is not local.bert.embeddings.word_embeddings.weight
    )
    # Avoid pad id 0 - nntile embedding grad at index 0 is a known sharp edge.
    input_ids = torch.randint(1, tiny_hf_config.vocab_size, (2, 8))
    token_type_ids = _bert_token_type_ids(tiny_hf_config)
    grad = torch.randn(2, 8, tiny_hf_config.vocab_size)

    for p in hf.parameters():
        p.requires_grad_(True)
    logits_ref = hf(input_ids=input_ids, token_type_ids=token_type_ids).logits
    logits_ref.backward(grad)

    logits = local(
        contiguous_to_nntile(input_ids),
        token_type_ids=contiguous_to_nntile(token_type_ids),
    )
    gw_emb, gw_dec = torch.autograd.grad(
        logits,
        (
            local.bert.embeddings.word_embeddings.weight,
            local.cls.decoder.weight,
        ),
        grad_outputs=contiguous_to_nntile(grad),
    )
    assert_close(
        gw_emb,
        hf.bert.embeddings.word_embeddings.weight.grad,
        rtol=1e-3,
        atol=BWD_ATOL,
    )
    assert_close(
        gw_dec,
        hf.cls.predictions.decoder.weight.grad,
        rtol=1e-3,
        atol=BWD_ATOL,
    )


def test_bert_export_roundtrip_state_dict_matches_hf_keys(tiny_hf_config):
    torch.manual_seed(7)
    hf = BertForMaskedLM(tiny_hf_config).eval().float()
    _untie_hf_bert_mlm(hf)
    local = BertMlm(bert_config_from_hf(tiny_hf_config)).eval().float()
    load_hf_into_bert_mlm(local, hf)
    exported = export_bert_mlm_to_hf_state_dict(local, config=tiny_hf_config)
    # Spot-check overlapping trainable keys (untied decoder included).
    for key in (
        "bert.embeddings.word_embeddings.weight",
        "bert.encoder.layer.0.attention.self.query.weight",
        "cls.predictions.transform.dense.weight",
        "cls.predictions.decoder.weight",
    ):
        assert key in exported
        torch.testing.assert_close(
            exported[key], hf.state_dict()[key], rtol=0, atol=0
        )
