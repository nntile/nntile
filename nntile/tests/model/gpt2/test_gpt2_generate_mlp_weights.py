#!/usr/bin/env python3
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file nntile/tests/model/gpt2/test_gpt2_generate_mlp_weights.py
# Regression: GPT-2 MLP weights match HF Conv1D / graph Linear layout.
#
# @version 1.1.0

"""Guard MLP weight conversion in ``generate_test_data.py``.

Graph ``Linear`` / ``Mlp`` expect C-order ``[out, in]`` weights. HuggingFace
GPT-2 ``Conv1D`` stores ``(in, out)``; transpose to match graph layout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_test_data import (  # noqa: E402
    _conv1d_to_linear_weight, _gpt2_mlp)


def test_mlp_weights_match_generate_test_data() -> None:
    torch.manual_seed(0)
    config = GPT2Config(
        n_embd=64,
        n_inner=128,
        n_head=4,
        n_layer=1,
        _attn_implementation="eager",
    )
    mlp = GPT2MLP(config.n_inner, config).eval()

    ref = _gpt2_mlp(mlp, "mlp")
    assert np.array_equal(
        ref["mlp.fc1.weight"], _conv1d_to_linear_weight(mlp.c_fc))
    assert np.array_equal(
        ref["mlp.fc2.weight"], _conv1d_to_linear_weight(mlp.c_proj))


def test_mlp_forward_parity_with_hf() -> None:
    torch.manual_seed(1)
    hidden, seq, batch = 64, 4, 2
    config = GPT2Config(
        n_embd=hidden,
        n_inner=128,
        n_head=4,
        n_layer=1,
        _attn_implementation="eager",
    )
    mlp = GPT2MLP(config.n_inner, config).eval()

    rng = np.random.default_rng(1)
    x = rng.standard_normal((batch, seq, hidden)).astype(np.float32)
    x_pt = torch.tensor(x.copy())
    with torch.no_grad():
        y_hf = mlp(x_pt).numpy()

    w1 = _conv1d_to_linear_weight(mlp.c_fc)
    w2 = _conv1d_to_linear_weight(mlp.c_proj)
    h = x @ w1.T
    h = F.gelu(torch.from_numpy(h), approximate="tanh").numpy()
    y_nt = h @ w2.T

    rel = float(np.linalg.norm(y_hf - y_nt) / np.linalg.norm(y_hf))
    assert rel < 1e-5, rel


def test_transpose_required_for_linear_layout() -> None:
    config = GPT2Config(n_embd=8, n_inner=16, n_head=2, n_layer=1)
    mlp = GPT2MLP(config.n_inner, config).eval()
    w = mlp.c_fc.weight.detach().numpy()
    linear_w = _conv1d_to_linear_weight(mlp.c_fc)
    assert linear_w.shape == (config.n_inner, config.n_embd)
    assert linear_w.shape == w.T.shape
    assert np.array_equal(linear_w, w.T)
    assert not np.array_equal(linear_w, w)


if __name__ == "__main__":
    test_mlp_weights_match_generate_test_data()
    test_mlp_forward_parity_with_hf()
    test_transpose_required_for_linear_layout()
    print("OK")
