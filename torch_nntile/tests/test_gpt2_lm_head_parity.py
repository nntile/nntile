# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_gpt2_lm_head_parity.py
# GPT2LMHead forward/backward parity vs HuggingFace GPT2LMHeadModel.

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("transformers")

import torch
from transformers import GPT2Config, GPT2LMHeadModel

import torch_nntile
from torch_nntile import _C
from conftest import nntile_cpu, subprocess_environ
from torch_nntile.models.gpt2_hf_loader import load_hf_into_gpt2_lm_head
from torch_nntile.models.gpt2_minimal import (
    GPT2Attention,
    GPT2Block,
    GPT2LMHead,
    GPT2MLP,
    GPT2Model,
    make_causal_sdpa_mask,
)


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)


@pytest.fixture
def tiny_gpt2_config() -> GPT2Config:
    config = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=64,
        n_positions=32,
        vocab_size=128,
        n_inner=256,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    return config


def _make_models(config: GPT2Config):
    torch.manual_seed(0)
    config.tie_word_embeddings = False
    hf = GPT2LMHeadModel(config).eval().float()
    minimal = GPT2LMHead(config).eval().float()
    load_hf_into_gpt2_lm_head(minimal, hf)
    minimal = minimal.to("nntile")
    return hf, minimal


def _assert_close(
    got: torch.Tensor,
    ref: torch.Tensor,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    torch.testing.assert_close(nntile_cpu(got), ref.cpu(), rtol=rtol, atol=atol)


def test_gpt2_model_forward_shape(tiny_gpt2_config):
    _, minimal = _make_models(tiny_gpt2_config)
    input_ids = torch.randint(0, tiny_gpt2_config.vocab_size, (2, 8)).to("nntile")
    hidden = minimal.transformer(input_ids)
    assert hidden.shape == (2, 8, tiny_gpt2_config.n_embd)
    assert hidden.dtype == torch.float32


def test_gpt2_lm_head_forward_shape(tiny_gpt2_config):
    _, minimal = _make_models(tiny_gpt2_config)
    input_ids = torch.randint(0, tiny_gpt2_config.vocab_size, (2, 8)).to("nntile")
    logits = minimal(input_ids)
    assert logits.shape == (2, 8, tiny_gpt2_config.vocab_size)
    assert logits.dtype == torch.float32


def test_gpt2_lm_head_forward_matches_hf(tiny_gpt2_config):
    hf, minimal = _make_models(tiny_gpt2_config)
    input_ids = torch.randint(0, tiny_gpt2_config.vocab_size, (2, 8)).to("nntile")
    with torch.no_grad():
        ref = hf(nntile_cpu(input_ids)).logits
        out = minimal(input_ids)
    _assert_close(out, ref)


def test_gpt2_lm_head_weights_are_untied(tiny_gpt2_config):
    torch.manual_seed(0)
    hf = GPT2LMHeadModel(tiny_gpt2_config).eval().float()
    minimal = GPT2LMHead(tiny_gpt2_config).eval().float()
    load_hf_into_gpt2_lm_head(minimal, hf)
    assert minimal.lm_head.weight is not minimal.transformer.wte.weight
    assert hf.lm_head.weight is not hf.transformer.wte.weight


def test_gpt2_block_forward_matches_hf(tiny_gpt2_config):
    torch.manual_seed(1)
    hf = GPT2LMHeadModel(tiny_gpt2_config).eval().float()
    block = GPT2Block(tiny_gpt2_config).eval().float()
    hf_block = hf.transformer.h[0]

    block.ln_1.load_state_dict(hf_block.ln_1.state_dict())
    block.ln_2.load_state_dict(hf_block.ln_2.state_dict())
    load_hf_into_gpt2_lm_head(
        GPT2LMHead(tiny_gpt2_config),
        hf,
    )
    from torch_nntile.models.gpt2_hf_loader import _split_hf_attn_weights

    hidden = tiny_gpt2_config.n_embd
    n_heads = tiny_gpt2_config.n_head
    head_size = hidden // n_heads
    attn_w = _split_hf_attn_weights(hf_block.attn, hidden, n_heads, head_size)
    block.attn.q_weight.data.copy_(attn_w["q_weight"])
    block.attn.k_weight.data.copy_(attn_w["k_weight"])
    block.attn.v_weight.data.copy_(attn_w["v_weight"])
    block.attn.o_weight.data.copy_(attn_w["o_weight"])
    block.attn.q_bias.data.copy_(attn_w["q_bias"])
    block.attn.k_bias.data.copy_(attn_w["k_bias"])
    block.attn.v_bias.data.copy_(attn_w["v_bias"])
    block.attn.o_bias.data.copy_(attn_w["o_bias"])
    block.mlp.c_fc.weight.data.copy_(hf_block.mlp.c_fc.weight.data.t())
    block.mlp.c_fc.bias.data.copy_(hf_block.mlp.c_fc.bias.data)
    block.mlp.c_proj.weight.data.copy_(hf_block.mlp.c_proj.weight.data.t())
    block.mlp.c_proj.bias.data.copy_(hf_block.mlp.c_proj.bias.data)

    block = block.to("nntile")
    x_cpu = torch.randn(2, 8, hidden)
    mask = make_causal_sdpa_mask(8).to("nntile")
    with torch.no_grad():
        ref = hf_block(x_cpu, attention_mask=None)[0]
        out = block(x_cpu.to("nntile"), mask)
    _assert_close(out, ref)


def test_gpt2_mlp_forward_matches_hf(tiny_gpt2_config):
    torch.manual_seed(2)
    from transformers.models.gpt2.modeling_gpt2 import GPT2MLP as HfMLP

    hf_mlp = HfMLP(tiny_gpt2_config.n_inner, tiny_gpt2_config).eval().float()
    mlp = GPT2MLP(tiny_gpt2_config).eval().float()
    mlp.c_fc.weight.data.copy_(hf_mlp.c_fc.weight.data.t())
    mlp.c_fc.bias.data.copy_(hf_mlp.c_fc.bias.data)
    mlp.c_proj.weight.data.copy_(hf_mlp.c_proj.weight.data.t())
    mlp.c_proj.bias.data.copy_(hf_mlp.c_proj.bias.data)
    mlp = mlp.to("nntile")

    x_cpu = torch.randn(2, 8, tiny_gpt2_config.n_embd)
    with torch.no_grad():
        ref = hf_mlp(x_cpu)
        out = mlp(x_cpu.to("nntile"))
    _assert_close(out, ref)


def test_gpt2_mlp_backward_matches_hf(tiny_gpt2_config):
    torch.manual_seed(3)
    from transformers.models.gpt2.modeling_gpt2 import GPT2MLP as HfMLP

    hf_mlp = HfMLP(tiny_gpt2_config.n_inner, tiny_gpt2_config).eval().float()
    mlp = GPT2MLP(tiny_gpt2_config).eval().float()
    mlp.c_fc.weight.data.copy_(hf_mlp.c_fc.weight.data.t())
    mlp.c_fc.bias.data.copy_(hf_mlp.c_fc.bias.data)
    mlp.c_proj.weight.data.copy_(hf_mlp.c_proj.weight.data.t())
    mlp.c_proj.bias.data.copy_(hf_mlp.c_proj.bias.data)
    mlp = mlp.to("nntile")

    x_cpu = torch.randn(2, 8, tiny_gpt2_config.n_embd, requires_grad=True)
    grad_out = torch.randn(2, 8, tiny_gpt2_config.n_embd)
    y_ref = hf_mlp(x_cpu)
    y_ref.backward(grad_out)

    x_nnt = x_cpu.detach().to("nntile").requires_grad_(True)
    params_nnt = [x_nnt, mlp.c_fc.weight, mlp.c_fc.bias, mlp.c_proj.weight, mlp.c_proj.bias]
    y = mlp(x_nnt)
    gx_nnt, gfc_w, gfc_b, gcp_w, gcp_b = torch.autograd.grad(
        y,
        params_nnt,
        grad_outputs=grad_out.to("nntile"),
    )

    _assert_close(gx_nnt, x_cpu.grad)
    _assert_close(gfc_w, hf_mlp.c_fc.weight.grad.t())
    _assert_close(gcp_w, hf_mlp.c_proj.weight.grad.t())


def test_gpt2_attention_forward_matches_hf(tiny_gpt2_config):
    torch.manual_seed(4)
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention as HfAttention

    hf_attn = HfAttention(tiny_gpt2_config, layer_idx=0).eval().float()
    attn = GPT2Attention(tiny_gpt2_config).eval().float()
    from torch_nntile.models.gpt2_hf_loader import _split_hf_attn_weights

    hidden = tiny_gpt2_config.n_embd
    n_heads = tiny_gpt2_config.n_head
    head_size = hidden // n_heads
    attn_w = _split_hf_attn_weights(hf_attn, hidden, n_heads, head_size)
    attn.q_weight.data.copy_(attn_w["q_weight"])
    attn.k_weight.data.copy_(attn_w["k_weight"])
    attn.v_weight.data.copy_(attn_w["v_weight"])
    attn.o_weight.data.copy_(attn_w["o_weight"])
    attn.q_bias.data.copy_(attn_w["q_bias"])
    attn.k_bias.data.copy_(attn_w["k_bias"])
    attn.v_bias.data.copy_(attn_w["v_bias"])
    attn.o_bias.data.copy_(attn_w["o_bias"])
    attn = attn.to("nntile")

    x_cpu = torch.randn(2, 8, hidden)
    mask = make_causal_sdpa_mask(8).to("nntile")
    with torch.no_grad():
        ref = hf_attn(x_cpu)[0]
        out = attn(x_cpu.to("nntile"), mask)
    _assert_close(out, ref)


def test_gpt2_attention_backward_matches_hf(tiny_gpt2_config):
    torch.manual_seed(5)
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention as HfAttention

    hf_attn = HfAttention(tiny_gpt2_config, layer_idx=0).eval().float()
    attn = GPT2Attention(tiny_gpt2_config).eval().float()
    from torch_nntile.models.gpt2_hf_loader import _split_hf_attn_weights

    hidden = tiny_gpt2_config.n_embd
    n_heads = tiny_gpt2_config.n_head
    head_size = hidden // n_heads
    attn_w = _split_hf_attn_weights(hf_attn, hidden, n_heads, head_size)
    attn.q_weight.data.copy_(attn_w["q_weight"])
    attn.k_weight.data.copy_(attn_w["k_weight"])
    attn.v_weight.data.copy_(attn_w["v_weight"])
    attn.o_weight.data.copy_(attn_w["o_weight"])
    attn.q_bias.data.copy_(attn_w["q_bias"])
    attn.k_bias.data.copy_(attn_w["k_bias"])
    attn.v_bias.data.copy_(attn_w["v_bias"])
    attn.o_bias.data.copy_(attn_w["o_bias"])

    for p in hf_attn.parameters():
        p.requires_grad_(True)
    for p in attn.parameters():
        p.requires_grad_(True)
    attn = attn.to("nntile")

    x_cpu = torch.randn(2, 8, hidden, requires_grad=True)
    grad_out = torch.randn(2, 8, hidden)
    mask = make_causal_sdpa_mask(8).to("nntile")

    y_ref = hf_attn(x_cpu)[0]
    y_ref.backward(grad_out)

    x_nnt = x_cpu.detach().to("nntile").requires_grad_(True)
    y = attn(x_nnt, mask)
    gx_nnt, = torch.autograd.grad(
        y,
        x_nnt,
        grad_outputs=grad_out.to("nntile"),
    )

    _assert_close(gx_nnt, x_cpu.grad, atol=1e-3)


def test_gpt2_lm_head_backward_matches_hf(tiny_gpt2_config):
    hf, minimal = _make_models(tiny_gpt2_config)
    for p in hf.parameters():
        p.requires_grad_(True)
    for p in minimal.parameters():
        p.requires_grad_(True)

    input_ids = torch.randint(0, tiny_gpt2_config.vocab_size, (2, 8)).to("nntile")
    grad_out = torch.randn(2, 8, tiny_gpt2_config.vocab_size)

    hf.zero_grad(set_to_none=True)
    hf(nntile_cpu(input_ids)).logits.backward(grad_out)

    minimal.zero_grad(set_to_none=True)
    logits = minimal(input_ids)
    glm_w, gwte, gwpe = torch.autograd.grad(
        logits,
        (
            minimal.lm_head.weight,
            minimal.transformer.wte.weight,
            minimal.transformer.wpe.weight,
        ),
        grad_outputs=grad_out.to("nntile"),
    )

    _assert_close(glm_w, hf.lm_head.weight.grad, atol=1e-3)
    _assert_close(gwte, hf.transformer.wte.weight.grad, atol=1e-3)
    _assert_close(gwpe, hf.transformer.wpe.weight.grad)


def test_gpt2_lm_head_forward_deferred(tiny_gpt2_config):
    import subprocess
    import sys
    import textwrap

    env = subprocess_environ()

    script = textwrap.dedent(
        """
        import torch
        import torch_nntile
        from transformers import GPT2Config, GPT2LMHeadModel
        from torch_nntile.models.gpt2_minimal import GPT2LMHead
        from torch_nntile.models.gpt2_hf_loader import load_hf_into_gpt2_lm_head

        torch_nntile.init_context(
            ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
        )
        torch_nntile.restrict_cpu()

        config = GPT2Config(
            n_layer=1, n_head=2, n_embd=32, n_positions=16,
            vocab_size=64, n_inner=128, attn_pdrop=0.0, resid_pdrop=0.0,
            tie_word_embeddings=False,
        )
        config._attn_implementation = "eager"
        torch.manual_seed(0)
        hf = GPT2LMHeadModel(config).eval().float()
        model = GPT2LMHead(config).eval().float()
        load_hf_into_gpt2_lm_head(model, hf)
        model = model.to("nntile")

        input_ids = torch.randint(0, config.vocab_size, (1, 4)).to("nntile")
        with torch.no_grad():
            ref = hf(input_ids.cpu()).logits
        assert torch_nntile.has_pending_graph() is False
        out = model(input_ids)
        assert torch_nntile.has_pending_graph()
        torch_nntile.compile_graph()
        torch_nntile.run()
        assert not torch_nntile.has_pending_graph()
        assert out.shape == ref.shape
        assert torch.allclose(out.cpu(), ref, rtol=1e-4, atol=1e-4)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"graph forward subprocess failed ({proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def test_position_ids_and_mask_are_cached(tiny_gpt2_config):
    model = GPT2LMHead(tiny_gpt2_config).eval().float().to("nntile")
    model.warm_sequence_caches(
        batch_sizes=[2],
        seq_len=8,
        device="nntile",
    )
    tr = model.transformer
    pos0 = tr._position_ids_cache[(2, 8)]
    mask0 = tr._causal_mask_cache[8]
    ids = torch.randint(0, tiny_gpt2_config.vocab_size, (2, 8)).to("nntile")
    with torch.no_grad():
        _ = model(ids)
        _ = model(ids)
    assert tr._position_ids_cache[(2, 8)] is pos0
    assert tr._causal_mask_cache[8] is mask0
    assert pos0.device.type == "nntile"
    assert mask0.device.type == "nntile"
