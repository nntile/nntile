# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_mlp_mixer_parity.py
# MLP-Mixer forward/backward parity: naive CPU torch vs device=nntile.

from __future__ import annotations

import pytest
import torch

from torch_nntile import _C
from torch_nntile.models.mlp_mixer import (
    MlpMixer,
    MlpMixerConfig,
    MlpMixerCpu,
    copy_cpu_weights_to_nntile,
)
from conftest import nntile_cpu


pytestmark = pytest.mark.skipif(
    not _C.has_libnntile(),
    reason="torch_nntile built without libnntile (set NNTILE_BUILD_DIR)",
)


def _tiny_config() -> MlpMixerConfig:
    return MlpMixerConfig(
        channel_dim=8,
        init_patch_dim=4,
        projected_patch_dim=4,
        num_mixer_layers=2,
        n_classes=3,
        layer_norm_epsilon=1e-5,
    )


def _cpu_to_nnt_param_names(num_layers: int) -> dict[str, str]:
    mapping = {
        "mixer_sequence.0.weight": "stem_weight",
        "classification.weight": "classifier_weight",
    }
    for i in range(num_layers):
        c = f"mixer_sequence.{i + 1}"
        n = f"blocks.{i}"
        mapping.update(
            {
                f"{c}.norm_1.weight": f"{n}.norm_1.weight",
                f"{c}.norm_1.bias": f"{n}.norm_1.bias",
                f"{c}.norm_2.weight": f"{n}.norm_2.weight",
                f"{c}.norm_2.bias": f"{n}.norm_2.bias",
                f"{c}.mlp_1.fn.0.weight": f"{n}.mlp_1.fc1_weight",
                f"{c}.mlp_1.fn.2.weight": f"{n}.mlp_1.fc2_weight",
                f"{c}.mlp_2.fn.0.weight": f"{n}.mlp_2.fc1_weight",
                f"{c}.mlp_2.fn.2.weight": f"{n}.mlp_2.fc2_weight",
            }
        )
    return mapping


def test_mlp_mixer_forward_matches_cpu():
    torch.manual_seed(0)
    cfg = _tiny_config()
    cpu = MlpMixerCpu(cfg)
    nnt = MlpMixer(cfg)
    copy_cpu_weights_to_nntile(cpu, nnt)

    x_cpu = torch.randn(cfg.channel_dim, 2, cfg.init_patch_dim)
    with torch.no_grad():
        y_cpu = cpu(x_cpu)
        nnt = nnt.to("nntile")
        y_nnt = nntile_cpu(nnt(x_cpu.to("nntile")))

    assert y_nnt.shape == y_cpu.shape
    assert torch.allclose(y_nnt, y_cpu, rtol=2e-5, atol=2e-5)


def test_mlp_mixer_backward_matches_cpu():
    torch.manual_seed(1)
    cfg = _tiny_config()
    cpu = MlpMixerCpu(cfg)
    nnt = MlpMixer(cfg)
    copy_cpu_weights_to_nntile(cpu, nnt)

    x_cpu = torch.randn(
        cfg.channel_dim,
        2,
        cfg.init_patch_dim,
        requires_grad=True,
    )
    y_cpu = cpu(x_cpu)
    grad_out = torch.randn_like(y_cpu)
    y_cpu.backward(grad_out)

    nnt = nnt.to("nntile")
    x_nnt = x_cpu.detach().to("nntile").requires_grad_(True)
    y_nnt = nnt(x_nnt)
    named = list(nnt.named_parameters())
    gx_nnt, *grads_nnt = torch.autograd.grad(
        y_nnt,
        (x_nnt, *[p for _, p in named]),
        grad_outputs=grad_out.to("nntile"),
    )

    assert torch.allclose(
        nntile_cpu(gx_nnt),
        x_cpu.grad,
        rtol=2e-5,
        atol=2e-5,
    )
    nnt_grad = {name: g for (name, _), g in zip(named, grads_nnt)}
    cpu_named = dict(cpu.named_parameters())
    for cname, nname in _cpu_to_nnt_param_names(cfg.num_mixer_layers).items():
        assert torch.allclose(
            nntile_cpu(nnt_grad[nname]),
            cpu_named[cname].grad,
            rtol=2e-5,
            atol=2e-5,
        ), cname
