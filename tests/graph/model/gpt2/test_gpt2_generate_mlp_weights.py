#!/usr/bin/env python3
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file tests/graph/model/gpt2/test_gpt2_generate_mlp_weights.py
# Regression: gpt2_generate MLP weights match HF Conv1D / graph Linear layout.
#
# @version 1.1.0

"""Guard MLP weight conversion used by examples/gpt2_generate.py.

Graph ``Linear`` / ``Mlp`` expect ``[input_dim, output_dim]`` Fortran-order
bytes. HuggingFace GPT-2 ``Conv1D`` weights already use ``(in, out)`` (e.g.
``c_fc.weight`` is ``(hidden_size, n_inner)``). Transposing would break parity
with ``tests/graph/model/gpt2/generate_test_data.py`` and C++ Gpt2MLP tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

# examples/gpt2_generate.py
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "examples"))
from gpt2_generate import _conv1d_to_nntile_linear_weight  # noqa: E402

# tests/graph/model/gpt2/generate_test_data.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_test_data import _conv1d_to_linear_weight, _gpt2_mlp  # noqa: E402


def _simulate_gpt2_mlp_graph(
    x_hsb: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
) -> np.ndarray:
    """Match Gpt2MLP transpose + Mlp GEMM (seq, batch, hidden) layout."""
    x_sbh = np.transpose(x_hsb, (1, 2, 0))
    h = x_sbh @ w1
    h = F.gelu(torch.from_numpy(h), approximate="tanh").numpy()
    y_sbh = h @ w2
    return np.transpose(y_sbh, (2, 0, 1))


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
    w1 = _conv1d_to_nntile_linear_weight(
        mlp.c_fc.weight.detach().numpy())
    w2 = _conv1d_to_nntile_linear_weight(
        mlp.c_proj.weight.detach().numpy())

    assert np.array_equal(ref["mlp.fc1.weight"], w1)
    assert np.array_equal(ref["mlp.fc2.weight"], w2)
    assert np.array_equal(ref["mlp.fc1.weight"], _conv1d_to_linear_weight(mlp.c_fc))
    assert np.array_equal(ref["mlp.fc2.weight"], _conv1d_to_linear_weight(mlp.c_proj))


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

    x_hsb = np.random.randn(hidden, seq, batch).astype(np.float32)
    x_pt = torch.tensor(x_hsb.transpose(2, 1, 0).copy())
    with torch.no_grad():
        y_hf = mlp(x_pt).numpy()

    flat1 = _conv1d_to_nntile_linear_weight(mlp.c_fc.weight.detach().numpy())
    flat2 = _conv1d_to_nntile_linear_weight(mlp.c_proj.weight.detach().numpy())
    w1 = np.frombuffer(flat1.tobytes(), dtype=np.float32).reshape(
        hidden, config.n_inner, order="F")
    w2 = np.frombuffer(flat2.tobytes(), dtype=np.float32).reshape(
        config.n_inner, hidden, order="F")

    y_nt = _simulate_gpt2_mlp_graph(x_hsb, w1, w2)
    rel = float(np.linalg.norm(y_hf - y_nt.transpose(2, 1, 0)) / np.linalg.norm(y_hf))
    assert rel < 1e-5, rel


def test_transpose_would_break_layout() -> None:
    config = GPT2Config(n_embd=8, n_inner=16, n_head=2, n_layer=1)
    mlp = GPT2MLP(config.n_inner, config).eval()
    w = mlp.c_fc.weight.detach().numpy()
    correct = _conv1d_to_nntile_linear_weight(w)
    wrong = _conv1d_to_nntile_linear_weight(w.T)
    assert correct.shape == w.shape
    assert wrong.shape == w.T.shape
    assert not np.array_equal(correct, wrong)


if __name__ == "__main__":
    test_mlp_weights_match_generate_test_data()
    test_mlp_forward_parity_with_hf()
    test_transpose_would_break_layout()
    print("OK")
