# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_dit.py
# DiT host helpers and HF Diffusers parity (classic kernels).

from __future__ import annotations

import math

import pytest
import torch
from classic_graph import assert_classic_graph
from conftest import nntile_cpu
from parity_helpers import assert_close, contiguous_to_nntile

import torch_nntile
from torch_nntile.models.dit import (
    DiT,
    DiTConfig,
    nchw_to_unpatchify_tokens,
    patchify_nchw,
    sincos_2d_pos_embed,
    timestep_embedding_table,
    unpatchify_nchw,
)

pytestmark = pytest.mark.skipif(
    not getattr(torch_nntile, "NNTILE_NATIVE_OPS", False),
    reason="classic nntile-native ops not built",
)

RTOL = 1e-4
ATOL = 1e-4
BWD_ATOL = 1e-3


def test_patchify_matches_unfold():
    torch.manual_seed(0)
    images = torch.randn(2, 3, 8, 8)
    tokens = patchify_nchw(images, 2)
    assert tokens.shape == (2, 16, 12)
    unfold = torch.nn.functional.unfold(images, kernel_size=2, stride=2)
    torch.testing.assert_close(tokens, unfold.transpose(1, 2))


def test_unpatchify_inverse():
    torch.manual_seed(0)
    images = torch.randn(2, 3, 8, 8)
    tokens = nchw_to_unpatchify_tokens(images, 2)
    got = unpatchify_nchw(
        tokens, patch_size=2, out_channels=3, grid_h=4, grid_w=4
    )
    torch.testing.assert_close(got, images)


def test_timestep_embedding_table_matches_fourier():
    dim = 256
    n = 8
    table = timestep_embedding_table(n, dim)
    timesteps = torch.arange(n, dtype=torch.float32)
    half = dim // 2
    exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32)
    exponent = exponent / (half - 1.0)
    args = timesteps[:, None] * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    emb = torch.cat([emb[:, half:], emb[:, :half]], dim=-1)
    torch.testing.assert_close(table, emb)


def test_sincos_2d_pos_embed_shape():
    pos = sincos_2d_pos_embed(16, 4)
    assert pos.shape == (16, 16)


def test_dit_nchw_on_nntile_raises():
    torch_nntile.reset_graph_session()
    cfg = DiTConfig(
        sample_size=8,
        patch_size=2,
        in_channels=3,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        num_embeds_ada_norm=4,
    )
    model = DiT(cfg).eval().float().to("nntile")
    images = torch.randn(2, 3, 8, 8).contiguous().to("nntile")
    t = torch.zeros(2, dtype=torch.long).contiguous().to("nntile")
    y = torch.zeros(2, dtype=torch.long).contiguous().to("nntile")
    with pytest.raises(ValueError, match="patchify NCHW"):
        model(images, t, y)


def _tiny_hf_dit():
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    hf = (
        diffusers.DiTTransformer2DModel(
            sample_size=8,
            patch_size=2,
            in_channels=3,
            out_channels=3,
            num_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            dropout=0.0,
            attention_bias=True,
            activation_fn="gelu-approximate",
            num_embeds_ada_norm=8,
            norm_type="ada_norm_zero",
            norm_elementwise_affine=False,
            norm_eps=1e-5,
        )
        .eval()
        .float()
    )
    for module in hf.modules():
        if hasattr(module, "dropout_prob"):
            module.dropout_prob = 0.0
    return hf


def test_dit_hf_forward_backward_parity():
    from torch_nntile.models.dit_hf_loader import (
        dit_config_from_hf,
        load_hf_into_dit,
    )

    hf = _tiny_hf_dit()
    local = DiT(dit_config_from_hf(hf.config)).eval().float()
    load_hf_into_dit(local, hf)
    local = local.to("nntile")

    torch.manual_seed(1)
    images = torch.randn(2, 3, 8, 8)
    timestep = torch.randint(0, 8, (2,), dtype=torch.long)
    labels = torch.randint(0, 8, (2,), dtype=torch.long)

    with torch.no_grad():
        ref = hf(
            images,
            timestep=timestep,
            class_labels=labels,
            return_dict=True,
        ).sample

    patches = patchify_nchw(images, 2)
    tokens = local(
        contiguous_to_nntile(patches),
        contiguous_to_nntile(timestep),
        contiguous_to_nntile(labels),
    )
    assert_classic_graph()
    got = unpatchify_nchw(
        nntile_cpu(tokens),
        patch_size=2,
        out_channels=3,
        grid_h=4,
        grid_w=4,
    )
    assert_close(got, ref, rtol=RTOL, atol=ATOL)

    images_ref = images.detach().clone().requires_grad_(True)
    y_ref = hf(
        images_ref,
        timestep=timestep,
        class_labels=labels,
        return_dict=True,
    ).sample
    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)

    patches_n = contiguous_to_nntile(patches.detach()).requires_grad_(True)
    tokens_n = local(
        patches_n,
        contiguous_to_nntile(timestep),
        contiguous_to_nntile(labels),
    )
    token_grad = nchw_to_unpatchify_tokens(grad, 2)
    gw = torch.autograd.grad(
        tokens_n,
        local.blocks[0].mlp.fc1.weight,
        grad_outputs=contiguous_to_nntile(token_grad),
    )[0]
    assert_classic_graph()
    assert_close(
        gw,
        hf.transformer_blocks[0].ff.net[0].proj.weight.grad,
        rtol=1e-3,
        atol=BWD_ATOL,
    )
