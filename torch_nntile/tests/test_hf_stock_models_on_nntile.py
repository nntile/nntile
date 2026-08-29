# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_hf_stock_models_on_nntile.py
# Stock HuggingFace torch.nn models on device="nntile" (torch-native aten).

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("transformers")

import torch
from classic_graph import assert_torch_native_graph
from conftest import nntile_cpu
from transformers import (
    BertConfig,
    BertForMaskedLM,
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoConfig,
    GPTNeoForCausalLM,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    RobertaConfig,
    RobertaForMaskedLM,
    T5Config,
    T5ForConditionalGeneration,
)
from transformers.models.roberta.modeling_roberta import (
    create_position_ids_from_input_ids,
)

import torch_nntile

pytestmark = pytest.mark.skipif(
    not getattr(torch_nntile, "TORCH_NATIVE_OPS", False),
    reason="torch-native aten ops not built",
)

RTOL = 1e-4
ATOL = 1e-4
BWD_ATOL = 1e-3


def _ones_mask(ids: torch.Tensor) -> torch.Tensor:
    """Dense ones mask (skips HF ``aten::index`` pad-token warning)."""
    return torch.ones_like(ids)


def _assert_fwd_bwd(
    ref: torch.nn.Module,
    model: torch.nn.Module,
    *,
    run_ref,
    run_nnt,
    param_ref: torch.Tensor,
    param_nnt: torch.Tensor,
) -> None:
    y_ref = run_ref()
    y = run_nnt()
    # Graph must be inspected before nntile_cpu compiles/runs it.
    assert_torch_native_graph()
    torch.testing.assert_close(
        nntile_cpu(y), y_ref.detach().cpu(), rtol=RTOL, atol=ATOL
    )

    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)
    (gw,) = torch.autograd.grad(y, param_nnt, grad.to(y.device))
    torch.testing.assert_close(
        nntile_cpu(gw), param_ref.grad, rtol=1e-3, atol=BWD_ATOL
    )


def test_stock_gpt2_forward_backward_on_nntile():
    torch.manual_seed(0)
    cfg = GPT2Config(
        n_layer=1,
        n_head=2,
        n_embd=64,
        n_positions=16,
        vocab_size=128,
        n_inner=128,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        tie_word_embeddings=False,
    )
    cfg._attn_implementation = "eager"
    ref = GPT2LMHeadModel(cfg).eval().float()
    model = GPT2LMHeadModel(cfg).eval().float()
    with torch.no_grad():
        model.load_state_dict(ref.state_dict())
        model = model.to("nntile")
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    _assert_fwd_bwd(
        ref,
        model,
        run_ref=lambda: ref(ids).logits,
        run_nnt=lambda: model(ids.to("nntile")).logits,
        param_ref=ref.transformer.wte.weight,
        param_nnt=model.transformer.wte.weight,
    )


def test_stock_llama_forward_backward_on_nntile():
    torch.manual_seed(1)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    cfg._attn_implementation = "eager"
    ref = LlamaForCausalLM(cfg).eval().float()
    model = LlamaForCausalLM(cfg).eval().float()
    with torch.no_grad():
        model.load_state_dict(ref.state_dict())
        model = model.to("nntile")
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    _assert_fwd_bwd(
        ref,
        model,
        run_ref=lambda: ref(ids).logits,
        run_nnt=lambda: model(ids.to("nntile")).logits,
        param_ref=ref.model.embed_tokens.weight,
        param_nnt=model.model.embed_tokens.weight,
    )


def test_stock_gpt_neo_forward_backward_on_nntile():
    torch.manual_seed(2)
    cfg = GPTNeoConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=16,
        attention_types=[[["global"], 1], [["local"], 1]],
        window_size=4,
        activation_function="gelu_new",
        attention_dropout=0.0,
        embed_dropout=0.0,
        resid_dropout=0.0,
    )
    cfg._attn_implementation = "eager"
    ref = GPTNeoForCausalLM(cfg).eval().float()
    model = GPTNeoForCausalLM(cfg).eval().float()
    with torch.no_grad():
        model.load_state_dict(ref.state_dict())
        model = model.to("nntile")
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    _assert_fwd_bwd(
        ref,
        model,
        run_ref=lambda: ref(ids).logits,
        run_nnt=lambda: model(ids.to("nntile")).logits,
        param_ref=ref.transformer.wte.weight,
        param_nnt=model.transformer.wte.weight,
    )


def test_stock_gpt_neox_forward_backward_on_nntile():
    torch.manual_seed(3)
    cfg = GPTNeoXConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=16,
        rotary_pct=0.25,
        rotary_emb_base=10000.0,
        attention_bias=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        hidden_act="gelu",
    )
    cfg._attn_implementation = "eager"
    ref = GPTNeoXForCausalLM(cfg).eval().float()
    model = GPTNeoXForCausalLM(cfg).eval().float()
    with torch.no_grad():
        model.load_state_dict(ref.state_dict())
        model = model.to("nntile")
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    head = getattr(ref, "embed_out", None) or ref.lm_head
    head_n = getattr(model, "embed_out", None) or model.lm_head
    _assert_fwd_bwd(
        ref,
        model,
        run_ref=lambda: ref(ids).logits,
        run_nnt=lambda: model(ids.to("nntile")).logits,
        param_ref=head.weight,
        param_nnt=head_n.weight,
    )


def test_stock_bert_forward_backward_on_nntile():
    torch.manual_seed(4)
    cfg = BertConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=16,
        type_vocab_size=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    cfg._attn_implementation = "eager"
    ref = BertForMaskedLM(cfg).eval().float()
    model = BertForMaskedLM(cfg).eval().float()
    with torch.no_grad():
        model.load_state_dict(ref.state_dict())
        model = model.to("nntile")
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    types = torch.zeros_like(ids)
    mask = _ones_mask(ids)
    _assert_fwd_bwd(
        ref,
        model,
        run_ref=lambda: ref(
            ids, attention_mask=mask, token_type_ids=types
        ).logits,
        run_nnt=lambda: model(
            ids.to("nntile"),
            attention_mask=mask.to("nntile"),
            token_type_ids=types.to("nntile"),
        ).logits,
        param_ref=ref.bert.embeddings.word_embeddings.weight,
        param_nnt=model.bert.embeddings.word_embeddings.weight,
    )


def test_stock_roberta_forward_backward_on_nntile():
    torch.manual_seed(5)
    cfg = RobertaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=18,
        pad_token_id=1,
        type_vocab_size=1,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    cfg._attn_implementation = "eager"
    ref = RobertaForMaskedLM(cfg).eval().float()
    model = RobertaForMaskedLM(cfg).eval().float()
    with torch.no_grad():
        model.load_state_dict(ref.state_dict())
        model = model.to("nntile")
    ids = torch.randint(4, cfg.vocab_size, (2, 8))
    mask = _ones_mask(ids)
    position_ids = create_position_ids_from_input_ids(ids, cfg.pad_token_id)
    _assert_fwd_bwd(
        ref,
        model,
        run_ref=lambda: ref(
            ids, attention_mask=mask, position_ids=position_ids
        ).logits,
        run_nnt=lambda: model(
            ids.to("nntile"),
            attention_mask=mask.to("nntile"),
            position_ids=position_ids.to("nntile"),
        ).logits,
        param_ref=ref.roberta.embeddings.word_embeddings.weight,
        param_nnt=model.roberta.embeddings.word_embeddings.weight,
    )


def test_stock_t5_forward_backward_on_nntile():
    torch.manual_seed(6)
    cfg = T5Config(
        vocab_size=128,
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        dropout_rate=0.0,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    cfg._attn_implementation = "eager"
    ref = T5ForConditionalGeneration(cfg).eval().float()
    model = T5ForConditionalGeneration(cfg).eval().float()
    with torch.no_grad():
        model.load_state_dict(ref.state_dict())
        model = model.to("nntile")
    enc = torch.randint(0, cfg.vocab_size, (2, 8))
    dec = torch.randint(0, cfg.vocab_size, (2, 8))
    # Float masks: HF adds them to a fp32 causal mask (int64 + fp32 is
    # not registered on nntile add).
    enc_mask = torch.ones(enc.shape, dtype=torch.float32)
    dec_mask = torch.ones(dec.shape, dtype=torch.float32)
    _assert_fwd_bwd(
        ref,
        model,
        run_ref=lambda: ref(
            input_ids=enc,
            attention_mask=enc_mask,
            decoder_input_ids=dec,
            decoder_attention_mask=dec_mask,
        ).logits,
        run_nnt=lambda: model(
            input_ids=enc.to("nntile"),
            attention_mask=enc_mask.to("nntile"),
            decoder_input_ids=dec.to("nntile"),
            decoder_attention_mask=dec_mask.to("nntile"),
        ).logits,
        param_ref=ref.shared.weight,
        param_nnt=model.shared.weight,
    )
