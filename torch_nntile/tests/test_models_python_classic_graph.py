# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_models_python_classic_graph.py
# Python torch_nntile.models (classic kernels) must record zero TORCH_*.

from __future__ import annotations

import pytest
import torch
from classic_graph import assert_classic_graph
from transformers import GPT2Config

import torch_nntile
from torch_nntile.models.bert import BertConfig, BertMlm
from torch_nntile.models.deep_relu import DeepReLU
from torch_nntile.models.gpt2_minimal import GPT2LMHead
from torch_nntile.models.gpt_neo import GPTNeoCausal, GPTNeoConfig
from torch_nntile.models.gpt_neox import GPTNeoXCausal, GPTNeoXConfig
from torch_nntile.models.llama import LlamaCausal, LlamaConfig
from torch_nntile.models.mlp_mixer import MlpMixer, MlpMixerConfig
from torch_nntile.models.roberta import RobertaConfig, RobertaMlm
from torch_nntile.models.t5 import T5Config, T5ForConditionalGeneration

pytestmark = pytest.mark.skipif(
    not getattr(torch_nntile, "NNTILE_NATIVE_OPS", False),
    reason="classic nntile-native ops not built",
)


def _ids(batch: int = 2, seq: int = 8, vocab: int = 128) -> torch.Tensor:
    return (
        torch.randint(0, vocab, (batch, seq), dtype=torch.long)
        .contiguous()
        .to("nntile")
    )


def _fwd_bwd_classic(out: torch.Tensor) -> None:
    assert out.device.type == "nntile"
    grad = torch.ones(tuple(out.shape), dtype=out.dtype).contiguous()
    out.backward(grad.to(out.device))
    assert_classic_graph()


@pytest.mark.parametrize("n_kv", [4, 2])
def test_python_llama_classic_graph_fwd_bwd(n_kv: int):
    torch_nntile.reset_graph_session()
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=n_kv,
        max_position_embeddings=16,
    )
    model = LlamaCausal(cfg).eval().float().to("nntile")
    out = model(_ids())
    _fwd_bwd_classic(out)


def test_python_gpt2_classic_graph_fwd_bwd():
    torch_nntile.reset_graph_session()
    cfg = GPT2Config(
        n_layer=1,
        n_head=4,
        n_embd=64,
        n_positions=16,
        vocab_size=128,
        n_inner=128,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        tie_word_embeddings=False,
    )
    model = GPT2LMHead(cfg).eval().float().to("nntile")
    out = model(_ids())
    _fwd_bwd_classic(out)


def test_python_gpt_neo_classic_graph_fwd_bwd():
    torch_nntile.reset_graph_session()
    cfg = GPTNeoConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=16,
        window_size=4,
    )
    model = GPTNeoCausal(cfg).eval().float().to("nntile")
    out = model(_ids())
    _fwd_bwd_classic(out)


def test_python_gpt_neox_classic_graph_fwd_bwd():
    torch_nntile.reset_graph_session()
    cfg = GPTNeoXConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=16,
        rotary_pct=0.25,
    )
    model = GPTNeoXCausal(cfg).eval().float().to("nntile")
    out = model(_ids())
    _fwd_bwd_classic(out)


def test_python_bert_classic_graph_fwd_bwd():
    torch_nntile.reset_graph_session()
    cfg = BertConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=16,
    )
    model = BertMlm(cfg).eval().float().to("nntile")
    ids = _ids()
    types = torch.zeros_like(ids.cpu()).contiguous().to("nntile")
    out = model(ids, token_type_ids=types)
    _fwd_bwd_classic(out)


def test_python_roberta_classic_graph_fwd_bwd():
    torch_nntile.reset_graph_session()
    cfg = RobertaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=18,
        pad_token_id=1,
    )
    model = RobertaMlm(cfg).eval().float().to("nntile")
    ids = torch.randint(4, 128, (2, 8), dtype=torch.long)
    ids[0, 0] = 1
    ids = ids.contiguous().to("nntile")
    types = torch.zeros_like(ids.cpu()).contiguous().to("nntile")
    out = model(ids, token_type_ids=types)
    _fwd_bwd_classic(out)


def test_python_t5_classic_graph_fwd_bwd():
    torch_nntile.reset_graph_session()
    cfg = T5Config(
        vocab_size=128,
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
    )
    model = T5ForConditionalGeneration(cfg).eval().float().to("nntile")
    out = model(_ids(), _ids())
    _fwd_bwd_classic(out)


def test_python_mixer_classic_graph_fwd_bwd():
    torch_nntile.reset_graph_session()
    cfg = MlpMixerConfig(
        channel_dim=8,
        init_patch_dim=4,
        projected_patch_dim=4,
        num_mixer_layers=1,
        n_classes=3,
    )
    model = MlpMixer(cfg).eval().float().to("nntile")
    x = torch.randn(8, 2, 4).contiguous().to("nntile")
    out = model(x)
    _fwd_bwd_classic(out)


def test_python_deep_relu_classic_graph_fwd_bwd():
    torch_nntile.reset_graph_session()
    model = DeepReLU(32, 64, 8, 2).eval().float().to("nntile")
    x = torch.randn(4, 32).contiguous().to("nntile")
    out = model(x)
    _fwd_bwd_classic(out)
