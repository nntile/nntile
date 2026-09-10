# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_mul_parity.py
# Parity tests for nntile mul via TensorGraph (torch-native path).

import torch
from conftest import nntile_cpu

import torch_nntile


def test_mul_matches_cpu():
    a_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b_cpu = torch.tensor([[0.5, -1.0], [2.0, 0.25]])

    a = a_cpu.to("nntile")
    b = b_cpu.to("nntile")
    z = a * b

    assert z.device.type == "nntile"
    assert torch.allclose(nntile_cpu(z), a_cpu * b_cpu)


def test_mul_inplace_broadcast_matches_cpu():
    a_cpu = torch.randn(2, 3, 4, dtype=torch.float32)
    b_cpu = torch.randn(1, 3, 4, dtype=torch.float32)
    a_nnt = a_cpu.clone().to("nntile")
    b_nnt = b_cpu.to("nntile")
    a_cpu.mul_(b_cpu)
    a_nnt.mul_(b_nnt)
    torch.testing.assert_close(nntile_cpu(a_nnt), a_cpu, rtol=1e-5, atol=1e-5)


def test_mul_out_of_place_broadcast_matches_cpu():
    a_cpu = torch.randn(2, 3, 4, dtype=torch.float32)
    b_cpu = torch.randn(1, 3, 4, dtype=torch.float32)
    ref = a_cpu * b_cpu
    got = a_cpu.to("nntile") * b_cpu.to("nntile")
    torch.testing.assert_close(nntile_cpu(got), ref, rtol=1e-5, atol=1e-5)


def test_mul_rope_style_broadcast_matches_cpu():
    # NeoX RoPE: q_rot [B,H,S,D] * cos [B,1,S,D] after unsqueeze.
    q_cpu = torch.randn(1, 4, 8, 16, dtype=torch.float32)
    cos_cpu = torch.randn(1, 1, 8, 16, dtype=torch.float32)
    ref = q_cpu * cos_cpu
    got = q_cpu.to("nntile") * cos_cpu.to("nntile")
    torch.testing.assert_close(nntile_cpu(got), ref, rtol=1e-5, atol=1e-5)


def test_mul_bool_inplace_broadcast_matches_cpu():
    a_cpu = torch.tensor([[True, False], [True, True]])
    b_cpu = torch.tensor([[True], [False]])
    a_nnt = a_cpu.clone().to("nntile")
    b_nnt = b_cpu.to("nntile")
    a_cpu.mul_(b_cpu)
    a_nnt.mul_(b_nnt)
    assert torch.equal(nntile_cpu(a_nnt), a_cpu)


def test_mul_fp32_inplace_bool_mask_matches_cpu():
    # GPT-Neo eager causal mask: float_mask *= bool_comparison
    mask_cpu = torch.full((4, 8), -1e4, dtype=torch.float32)
    pred_cpu = torch.tensor([[True, True, False, False, True, True, False, False]])
    mask_cpu.mul_(pred_cpu)
    mask_nnt = mask_cpu.clone().to("nntile")
    pred_nnt = pred_cpu.to("nntile")
    mask_nnt.mul_(pred_nnt)
    torch.testing.assert_close(nntile_cpu(mask_nnt), mask_cpu, rtol=0, atol=0)


def test_mul_fp32_bool_out_of_place_broadcast_matches_cpu():
    # Out-of-place float * bool via at::mul_out (StarPU iargs[15]=3).
    mask_cpu = torch.full((4, 8), -1e4, dtype=torch.float32)
    pred_cpu = torch.tensor([[True, True, False, False, True, True, False, False]])
    ref = mask_cpu * pred_cpu
    got = mask_cpu.to("nntile") * pred_cpu.to("nntile")
    torch.testing.assert_close(nntile_cpu(got), ref, rtol=0, atol=0)
