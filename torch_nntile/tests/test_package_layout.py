# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_package_layout.py
# Public torch_nntile.nn.functional / module / model import layout.

from __future__ import annotations

import torch_nntile
from torch_nntile.nn import functional, module
from torch_nntile.nn.model import DeepReLU


def test_nn_submodules_are_exported():
    assert torch_nntile.nn.functional is functional
    assert torch_nntile.nn.module is module
    assert callable(functional.relu)
    assert callable(functional.gemm)
    assert torch_nntile.nn.Linear is module.Linear
    assert torch_nntile.nn.ReLU is module.ReLU


def test_nntile_native_model_import():
    assert DeepReLU is torch_nntile.nn.model.DeepReLU
    model = DeepReLU.tiny()
    assert any(isinstance(m, torch_nntile.nn.Linear) for m in model.modules())


def test_models_compat_alias():
    from torch_nntile.models import DeepReLU as CompatDeepReLU
    from torch_nntile.models.deep_relu import DeepReLU as CompatSubmodule

    assert CompatDeepReLU is DeepReLU
    assert CompatSubmodule is DeepReLU
