# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_models_python_tiled_smoke.py
# One tiled classic-kernel smoke per Python model family.

"""Each ``torch_nntile.models`` family must compile with axis-group tiling.

Cached RoPE / position / token-type tables are independent uploads. Name
their batch axis too, or the tiled activations disagree with untiled
tables (``grid_linear`` OOB).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from conftest import subprocess_environ

import torch_nntile

pytestmark = pytest.mark.skipif(
    not getattr(torch_nntile, "NNTILE_NATIVE_OPS", False),
    reason="classic nntile-native ops not built",
)

_TESTS_DIR = Path(__file__).resolve().parent

_PREAMBLE = """
import torch
import torch_nntile
from classic_graph import assert_classic_graph

torch_nntile.init_context(
    ncpu=1, ncuda=0, verbose=0, cpu_fallback=False
)
torch_nntile.restrict_cpu()
"""


def _run(body: str) -> None:
    env = subprocess_environ()
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{_TESTS_DIR}:{py_path}" if py_path else str(_TESTS_DIR)
    )
    script = _PREAMBLE + "\n" + textwrap.dedent(body)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"subprocess failed ({proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def test_tiled_deep_relu():
    _run(
        """
        from torch_nntile.models.deep_relu import DeepReLU
        model = DeepReLU(32, 64, 8, 2).eval().float().to("nntile")
        ids = torch.randn(4, 32).contiguous().to("nntile")
        out = model(ids)
        torch_nntile.set_axis_group_name(ids, {0: "batch"})
        torch_nntile.set_axis_group_name(out, {0: "batch"})
        torch_nntile.set_axis_group_tiling("batch", [2, 2])
        torch_nntile.execute()
        assert_classic_graph()
        assert tuple(out.detach().cpu().shape) == (4, 8)
        """
    )


def test_tiled_gpt2():
    _run(
        """
        from transformers import GPT2Config
        from torch_nntile.models.gpt2_minimal import GPT2LMHead
        cfg = GPT2Config(
            n_layer=1, n_head=4, n_embd=64, n_positions=16,
            vocab_size=128, n_inner=128, attn_pdrop=0.0,
            resid_pdrop=0.0, embd_pdrop=0.0, tie_word_embeddings=False,
        )
        model = GPT2LMHead(cfg).eval().float().to("nntile")
        ids = torch.randint(0, 128, (2, 8), dtype=torch.long)
        ids = ids.contiguous().to("nntile")
        out = model(ids)
        torch_nntile.set_axis_group_name(ids, {0: "batch"})
        torch_nntile.set_axis_group_name(out, {0: "batch"})
        torch_nntile.set_axis_group_tiling("batch", [1, 1])
        info = torch_nntile.format_axis_groups()
        assert "name='batch'" in info
        assert "pending_tile=1,1" in info
        torch_nntile.execute()
        assert_classic_graph()
        _ = out.detach().cpu()
        """
    )


def test_tiled_llama():
    _run(
        """
        from torch_nntile.models.llama import LlamaCausal, LlamaConfig
        cfg = LlamaConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=1, num_attention_heads=4,
            num_key_value_heads=4, max_position_embeddings=16,
        )
        model = LlamaCausal(cfg).eval().float().to("nntile")
        ids = torch.randint(0, 128, (2, 8), dtype=torch.long)
        ids = ids.contiguous().to("nntile")
        out = model(ids)
        pos = model.model._position_ids_cache[(2, 8)]
        sin, cos = model.model._rope_cache[(2, 8)]
        for t in (ids, out, pos, sin, cos):
            torch_nntile.set_axis_group_name(t, {0: "batch"})
        torch_nntile.set_axis_group_tiling("batch", [1, 1])
        info = torch_nntile.format_axis_groups()
        assert "name='batch'" in info
        assert "pending_tile=1,1" in info
        torch_nntile.execute()
        assert_classic_graph()
        _ = out.detach().cpu()
        """
    )


def test_tiled_gpt_neo():
    _run(
        """
        from torch_nntile.models.gpt_neo import GPTNeoCausal, GPTNeoConfig
        cfg = GPTNeoConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4,
            max_position_embeddings=16, window_size=4,
        )
        model = GPTNeoCausal(cfg).eval().float().to("nntile")
        ids = torch.randint(0, 128, (2, 8), dtype=torch.long)
        ids = ids.contiguous().to("nntile")
        out = model(ids)
        torch_nntile.set_axis_group_name(ids, {0: "batch"})
        torch_nntile.set_axis_group_name(out, {0: "batch"})
        torch_nntile.set_axis_group_tiling("batch", [1, 1])
        info = torch_nntile.format_axis_groups()
        assert "name='batch'" in info
        assert "pending_tile=1,1" in info
        torch_nntile.execute()
        assert_classic_graph()
        _ = out.detach().cpu()
        """
    )


def test_tiled_gpt_neox():
    _run(
        """
        from torch_nntile.models.gpt_neox import GPTNeoXCausal, GPTNeoXConfig
        cfg = GPTNeoXConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=1, num_attention_heads=4,
            max_position_embeddings=16, rotary_pct=0.25,
        )
        model = GPTNeoXCausal(cfg).eval().float().to("nntile")
        ids = torch.randint(0, 128, (2, 8), dtype=torch.long)
        ids = ids.contiguous().to("nntile")
        out = model(ids)
        pos = model.gpt_neox._position_ids_cache[(2, 8)]
        sin, cos = model.gpt_neox._rope_cache[(2, 8)]
        for t in (ids, out, pos, sin, cos):
            torch_nntile.set_axis_group_name(t, {0: "batch"})
        torch_nntile.set_axis_group_tiling("batch", [1, 1])
        info = torch_nntile.format_axis_groups()
        assert "name='batch'" in info
        assert "pending_tile=1,1" in info
        torch_nntile.execute()
        assert_classic_graph()
        _ = out.detach().cpu()
        """
    )


def test_tiled_bert():
    _run(
        """
        from torch_nntile.models.bert import BertConfig, BertMlm
        cfg = BertConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=1, num_attention_heads=4,
            max_position_embeddings=16,
        )
        model = BertMlm(cfg).eval().float().to("nntile")
        ids = torch.randint(0, 128, (2, 8), dtype=torch.long)
        ids = ids.contiguous().to("nntile")
        types = torch.zeros(2, 8, dtype=torch.long).contiguous().to("nntile")
        pos = (
            torch.arange(8, dtype=torch.long)
            .unsqueeze(0)
            .expand(2, 8)
            .contiguous()
            .to("nntile")
        )
        out = model(ids, token_type_ids=types, position_ids=pos)
        for t in (ids, types, pos, out):
            torch_nntile.set_axis_group_name(t, {0: "batch"})
        torch_nntile.set_axis_group_tiling("batch", [1, 1])
        info = torch_nntile.format_axis_groups()
        assert "name='batch'" in info
        assert "pending_tile=1,1" in info
        torch_nntile.execute()
        assert_classic_graph()
        _ = out.detach().cpu()
        """
    )


def test_tiled_roberta():
    _run(
        """
        from torch_nntile.models.roberta import RobertaConfig, RobertaMlm
        cfg = RobertaConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=1, num_attention_heads=4,
            max_position_embeddings=18, pad_token_id=1,
        )
        model = RobertaMlm(cfg).eval().float().to("nntile")
        ids = torch.randint(4, 128, (2, 8), dtype=torch.long)
        ids[0, 0] = 1
        ids = ids.contiguous().to("nntile")
        types = torch.zeros(2, 8, dtype=torch.long).contiguous().to("nntile")
        pos = (
            torch.arange(8, dtype=torch.long)
            .unsqueeze(0)
            .expand(2, 8)
            .contiguous()
            .to("nntile")
        )
        out = model(ids, token_type_ids=types, position_ids=pos)
        for t in (ids, types, pos, out):
            torch_nntile.set_axis_group_name(t, {0: "batch"})
        torch_nntile.set_axis_group_tiling("batch", [1, 1])
        info = torch_nntile.format_axis_groups()
        assert "name='batch'" in info
        assert "pending_tile=1,1" in info
        torch_nntile.execute()
        assert_classic_graph()
        _ = out.detach().cpu()
        """
    )


def test_tiled_t5():
    _run(
        """
        from torch_nntile.models.t5 import T5Config, T5ForConditionalGeneration
        cfg = T5Config(
            vocab_size=128, d_model=64, d_kv=16, d_ff=128,
            num_layers=1, num_decoder_layers=1, num_heads=4,
        )
        model = T5ForConditionalGeneration(cfg).eval().float().to("nntile")
        ids = torch.randint(0, 128, (2, 8), dtype=torch.long)
        ids = ids.contiguous().to("nntile")
        dec = torch.randint(0, 128, (2, 8), dtype=torch.long)
        dec = dec.contiguous().to("nntile")
        out = model(ids, dec)
        torch_nntile.set_axis_group_name(ids, {0: "batch"})
        torch_nntile.set_axis_group_name(dec, {0: "batch"})
        torch_nntile.set_axis_group_name(out, {0: "batch"})
        torch_nntile.set_axis_group_tiling("batch", [1, 1])
        torch_nntile.execute()
        assert_classic_graph()
        _ = out.detach().cpu()
        """
    )


def test_tiled_mlp_mixer():
    _run(
        """
        from torch_nntile.models.mlp_mixer import MlpMixer, MlpMixerConfig
        cfg = MlpMixerConfig(
            channel_dim=8, init_patch_dim=4, projected_patch_dim=4,
            num_mixer_layers=1, n_classes=3,
        )
        model = MlpMixer(cfg).eval().float().to("nntile")
        ids = torch.randn(8, 2, 4).contiguous().to("nntile")
        out = model(ids)
        torch_nntile.set_axis_group_name(ids, {1: "batch"})
        torch_nntile.set_axis_group_name(out, {0: "batch"})
        torch_nntile.set_axis_group_tiling("batch", [1, 1])
        torch_nntile.execute()
        assert_classic_graph()
        _ = out.detach().cpu()
        """
    )
