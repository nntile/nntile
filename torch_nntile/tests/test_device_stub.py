# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_device_stub.py
# Tests for the PyTorch nntile device stub.

import torch
import pytest

import torch_nntile
from torch_nntile import _C, device


def test_import_registers_device():
    assert torch_nntile._registered
    assert device.type == "nntile"
    assert _C.is_registered()
    assert torch.device("nntile").type == "nntile"
    assert hasattr(torch, "nntile")


def test_empty_on_device():
    x = torch.empty((2, 3), device="nntile")
    assert x.device.type == "nntile"
    assert x.shape == (2, 3)
    assert x.dtype == torch.float32
    assert _C.buffer_nbytes(x) == x.element_size() * x.numel()


def test_cpu_to_nntile_copy():
    cpu = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    nnt = cpu.to("nntile")
    assert nnt.device.type == "nntile"
    assert _C.buffer_equal_cpu(nnt, cpu)


def test_nntile_to_cpu_copy():
    cpu = torch.tensor([1.0, -2.0, 3.5, 0.0])
    nnt = cpu.to("nntile")
    back = nnt.cpu()
    assert back.device.type == "cpu"
    assert torch.allclose(back, cpu)


def test_tensor_factory_on_device():
    x = torch.tensor([1.0, 2.0, 3.0], device="nntile")
    y = x.cpu()
    assert torch.allclose(y, torch.tensor([1.0, 2.0, 3.0]))
