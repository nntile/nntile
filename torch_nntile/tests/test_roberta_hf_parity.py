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
from conftest import nntile_cpu
from parity_helpers import assert_close, contiguous_to_nntile, copy_linear
from torch_nntile.models.bert import (
    BertAttention,
    BertIntermediate,
    BertLayer,
    BertSelfAttention,
)
from torch_nntile.models.roberta import (
    RobertaConfig,
    RobertaEmbeddings,
    RobertaMlm,
    RobertaMlmHead,
)
from torch_nntile.models.roberta_hf_loader import (
    export_roberta_mlm_to_hf_state_dict,
    load_hf_into_roberta_mlm,
    roberta_config_from_hf,
)
from torch_nntile.nn.linear import (
    linear_to_output_weight,
    linear_to_qkv_bias,
    linear_to_qkv_weight,
    qkv_to_linear_weight,
)
from transformers import RobertaConfig as HfRobertaConfig
from transformers import RobertaForMaskedLM
from transformers.models.roberta.modeling_roberta import (
    RobertaAttention as HfAttention,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaEmbeddings as HfEmbeddings,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaIntermediate as HfIntermediate,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaLayer as HfLayer,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaSelfAttention as HfSelfAttention,
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


def _untie_hf_roberta_mlm(hf: RobertaForMaskedLM) -> None:
    """Clone LM-head decoder so HF reference matches local untied grads."""
    head = hf.lm_head
    emb_w = hf.roberta.embeddings.word_embeddings.weight
    if head.decoder.weight.data_ptr() == emb_w.data_ptr():
        head.decoder.weight = torch.nn.Parameter(emb_w.detach().clone())


def _make_models(
    hf_cfg: HfRobertaConfig,
) -> tuple[RobertaForMaskedLM, RobertaMlm]:
    torch.manual_seed(0)
    hf = RobertaForMaskedLM(hf_cfg).eval().float()
    _untie_hf_roberta_mlm(hf)
    local = RobertaMlm(roberta_config_from_hf(hf_cfg)).eval().float()
    load_hf_into_roberta_mlm(local, hf)
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


@pytest.mark.parametrize("hidden_act", ["gelu", "gelu_new", "relu"])
def test_roberta_intermediate_activation_variants(hidden_act, tiny_hf_config):
    tiny_hf_config.hidden_act = hidden_act
    torch.manual_seed(2)
    hf_inter = HfIntermediate(tiny_hf_config).eval().float()
    local = (
        BertIntermediate(
            roberta_config_from_hf(tiny_hf_config).to_bert_config()
        )
        .eval()
        .float()
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


def test_roberta_embeddings_forward_with_pads_matches_hf(tiny_hf_config):
    torch.manual_seed(1)
    hf_emb = HfEmbeddings(tiny_hf_config).eval().float()
    local = (
        RobertaEmbeddings(roberta_config_from_hf(tiny_hf_config))
        .eval()
        .float()
    )
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


def test_roberta_intermediate_forward_backward_matches_hf(tiny_hf_config):
    torch.manual_seed(2)
    hf_inter = HfIntermediate(tiny_hf_config).eval().float()
    local = (
        BertIntermediate(
            roberta_config_from_hf(tiny_hf_config).to_bert_config()
        )
        .eval()
        .float()
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


def test_roberta_self_attention_forward_backward_matches_hf(tiny_hf_config):
    """Self-attn returns SDPA layout; HF merge is covered by BertAttention."""
    torch.manual_seed(3)
    hf_self = HfSelfAttention(tiny_hf_config).eval().float()
    local = (
        BertSelfAttention(
            roberta_config_from_hf(tiny_hf_config).to_bert_config()
        )
        .eval()
        .float()
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


def test_roberta_attention_forward_backward_matches_hf(tiny_hf_config):
    torch.manual_seed(4)
    hf_attn = HfAttention(tiny_hf_config).eval().float()
    local = (
        BertAttention(roberta_config_from_hf(tiny_hf_config).to_bert_config())
        .eval()
        .float()
    )
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


def test_roberta_layer_forward_backward_matches_hf(tiny_hf_config):
    torch.manual_seed(5)
    hf_layer = HfLayer(tiny_hf_config).eval().float()
    local = (
        BertLayer(roberta_config_from_hf(tiny_hf_config).to_bert_config())
        .eval()
        .float()
    )
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


def test_roberta_model_hidden_forward_backward_matches_hf(tiny_hf_config):
    hf, local = _make_models(tiny_hf_config)
    input_ids = _roberta_input_ids(tiny_hf_config)
    token_type_ids = _roberta_token_type_ids(tiny_hf_config)
    with torch.no_grad():
        ref = hf.roberta(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        ).last_hidden_state
        out = local.roberta(
            contiguous_to_nntile(input_ids),
            token_type_ids=contiguous_to_nntile(token_type_ids),
        )
    assert_close(out, ref, rtol=RTOL, atol=ATTN_ATOL)

    for p in hf.parameters():
        p.requires_grad_(True)
    for p in local.parameters():
        p.requires_grad_(True)
    grad = torch.randn_like(ref)
    y_ref = hf.roberta(
        input_ids=input_ids, token_type_ids=token_type_ids
    ).last_hidden_state
    y_ref.backward(grad)
    y = local.roberta(
        contiguous_to_nntile(input_ids),
        token_type_ids=contiguous_to_nntile(token_type_ids),
    )
    (gw,) = torch.autograd.grad(
        y,
        local.roberta.encoder.layer[0].attention.self.query.weight,
        contiguous_to_nntile(grad),
    )
    assert_close(
        _qkv_weight_grad_to_linear(gw),
        hf.roberta.encoder.layer[0].attention.self.query.weight.grad,
        rtol=1e-3,
        atol=BWD_ATOL,
    )


def test_roberta_mlm_head_forward_backward_matches_hf(tiny_hf_config):
    torch.manual_seed(6)
    hf, _ = _make_models(tiny_hf_config)
    cfg = roberta_config_from_hf(tiny_hf_config)
    head = RobertaMlmHead(cfg).eval().float()
    hf_head = hf.lm_head
    head.dense.weight.data.copy_(hf_head.dense.weight.data)
    head.dense.bias.data.copy_(hf_head.dense.bias.data)
    head.layer_norm.load_state_dict(hf_head.layer_norm.state_dict())
    head.decoder.weight.data.copy_(hf_head.decoder.weight.data)
    if hf_head.bias is not None:
        head.decoder.bias.data.copy_(hf_head.bias.data)
    elif hf_head.decoder.bias is not None:
        head.decoder.bias.data.copy_(hf_head.decoder.bias.data)
    head = head.to("nntile")

    x = torch.randn(2, 8, tiny_hf_config.hidden_size, requires_grad=True)
    y_ref = hf_head(x)
    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)
    x_n = contiguous_to_nntile(x.detach()).requires_grad_(True)
    y = head(x_n)
    assert_close(y, y_ref.detach(), rtol=RTOL, atol=ATTN_ATOL)
    (gx,) = torch.autograd.grad(y, x_n, contiguous_to_nntile(grad))
    assert_close(gx, x.grad, rtol=1e-3, atol=BWD_ATOL)


def test_roberta_mlm_loader_forward_matches_hf(tiny_hf_config):
    hf, local = _make_models(tiny_hf_config)
    input_ids = _roberta_input_ids(tiny_hf_config)
    token_type_ids = _roberta_token_type_ids(tiny_hf_config)
    with torch.no_grad():
        ref = hf(input_ids=input_ids, token_type_ids=token_type_ids).logits
        out = local(
            contiguous_to_nntile(input_ids),
            token_type_ids=contiguous_to_nntile(token_type_ids),
        )
    assert_close(out, ref, rtol=RTOL, atol=ATTN_ATOL)


def test_roberta_mlm_logits_forward_backward_query_weight_matches_hf(
    tiny_hf_config,
):
    hf, local = _make_models(tiny_hf_config)
    input_ids = _roberta_input_ids(tiny_hf_config)
    token_type_ids = _roberta_token_type_ids(tiny_hf_config)
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
        local.roberta.encoder.layer[0].attention.self.query.weight,
        grad_outputs=contiguous_to_nntile(grad),
    )
    assert_close(
        _qkv_weight_grad_to_linear(gw),
        hf.roberta.encoder.layer[0].attention.self.query.weight.grad,
        rtol=1e-3,
        atol=BWD_ATOL,
    )


def test_roberta_export_roundtrip_state_dict_matches_hf_keys(tiny_hf_config):
    torch.manual_seed(7)
    hf = RobertaForMaskedLM(tiny_hf_config).eval().float()
    _untie_hf_roberta_mlm(hf)
    local = RobertaMlm(roberta_config_from_hf(tiny_hf_config)).eval().float()
    load_hf_into_roberta_mlm(local, hf)
    exported = export_roberta_mlm_to_hf_state_dict(
        local, config=tiny_hf_config
    )
    for key in (
        "roberta.embeddings.word_embeddings.weight",
        "roberta.encoder.layer.0.attention.self.query.weight",
        "lm_head.dense.weight",
        "lm_head.decoder.weight",
    ):
        assert key in exported
        torch.testing.assert_close(
            exported[key], hf.state_dict()[key], rtol=0, atol=0
        )


def test_roberta_vs_bert_default_special_tokens_differ():
    """RoBERTa pad/bos/eos differ from Bert - configs must not be mixed."""
    from transformers import BertConfig

    r = HfRobertaConfig()
    b = BertConfig()
    assert r.pad_token_id == 1
    assert b.pad_token_id == 0
    assert r.bos_token_id == 0
    assert r.eos_token_id == 2
