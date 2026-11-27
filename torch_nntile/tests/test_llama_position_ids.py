# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/test_llama_position_ids.py
# LlamaModel must honor the contents of caller-supplied position_ids
# when building its RoPE tables (KV-cache offsets, packed sequences).
#
# The custom positions are deliberately NON-uniform (two packed
# segments): RoPE attention is shift-invariant, so a uniform offset of
# every position cannot distinguish correct tables from the arange
# default, while packed segments can.
#
# Each run owns a fresh graph session and releases every nntile tensor
# before the next reset (reusing tensors across sessions is not
# supported), and models are re-seeded per run so weights match.

import torch
from conftest import nntile_cpu
from torch_nntile.nn.model.llama import LlamaConfig, LlamaModel
from torch_nntile.rope import rope_sin_cos_from_position_ids

import torch_nntile


def small_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
    )


def packed_position_ids(batch: int, device: str) -> torch.Tensor:
    """Two segments: [0..3] then a jump to [100..103]."""
    seq = 8
    row = torch.tensor([0, 1, 2, 3, 100, 101, 102, 103])
    return row.unsqueeze(0).expand(batch, seq).contiguous().to(device)


def run_case(
    cfg: LlamaConfig,
    seed: int,
    input_ids: torch.Tensor,
    *,
    sin: torch.Tensor | None = None,
    cos: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """One forward in one fresh, fully released graph session."""
    torch.manual_seed(seed)
    torch_nntile.reset_graph_session()
    kwargs = {}
    if sin is not None:
        kwargs["sin"] = sin.to("nntile")
        kwargs["cos"] = cos.to("nntile")
    if position_ids is not None:
        kwargs["position_ids"] = position_ids.to("nntile")
    model = LlamaModel(cfg).eval().float().to("nntile")
    ids = input_ids.to("nntile")
    with torch.no_grad():
        out = model(ids, **kwargs)
    out_cpu = nntile_cpu(out)
    del out, model, ids, kwargs
    return out_cpu


def test_forward_with_packed_position_ids_matches_explicit_tables():
    """position_ids contents must match sin/cos for those positions."""
    torch.manual_seed(7)
    cfg = small_config()
    batch = 2
    input_ids = torch.randint(0, cfg.vocab_size, (batch, 8))
    packed_cpu = packed_position_ids(batch, "cpu")
    sin_ref, cos_ref = rope_sin_cos_from_position_ids(
        packed_cpu, cfg.head_dim, rope_theta=cfg.rope_theta
    )

    out_ref = run_case(cfg, 42, input_ids, sin=sin_ref, cos=cos_ref)
    out_act = run_case(cfg, 42, input_ids, position_ids=packed_cpu)
    assert torch.allclose(out_ref, out_act, rtol=1e-4, atol=1e-4)


def test_packed_positions_get_different_tables_than_arange():
    """Explicit ids must not be served the warmed arange table."""
    torch.manual_seed(8)
    cfg = small_config()
    batch = 1
    input_ids = torch.randint(0, cfg.vocab_size, (batch, 8))
    packed_cpu = packed_position_ids(batch, "cpu")
    sin_packed, cos_packed = rope_sin_cos_from_position_ids(
        packed_cpu, cfg.head_dim, rope_theta=cfg.rope_theta
    )
    sin_arn, cos_arn = rope_sin_cos_from_position_ids(
        torch.arange(8).unsqueeze(0).expand(batch, 8),
        cfg.head_dim,
        rope_theta=cfg.rope_theta,
    )

    # Packed-segment positions have a genuinely different relative
    # structure than arange, so both the tables and the model outputs
    # must differ.
    assert not torch.allclose(sin_packed, sin_arn, rtol=1e-3, atol=1e-4)
    out_packed = run_case(
        cfg, 43, input_ids, sin=sin_packed, cos=cos_packed
    )
    out_arn = run_case(cfg, 43, input_ids, sin=sin_arn, cos=cos_arn)
    assert not torch.allclose(out_packed, out_arn, rtol=1e-3, atol=1e-4)

    # The model's own table builder must agree with the reference for
    # the explicit ids (and not return the arange table).
    torch.manual_seed(43)
    torch_nntile.reset_graph_session()
    model = LlamaModel(cfg).eval().float().to("nntile")
    ids_nnt = input_ids.to("nntile")
    with torch.no_grad():
        _ = nntile_cpu(model(ids_nnt))  # warm the (batch, seq) cache
    sin_built, cos_built = model._rope_tables_from_position_ids(
        packed_cpu.to("nntile")
    )
    assert not torch.allclose(
        sin_built.cpu(), sin_arn, rtol=1e-3, atol=1e-4
    )
    assert torch.allclose(
        sin_built.cpu(), sin_packed, rtol=1e-5, atol=1e-6
    )
    assert torch.allclose(
        cos_built.cpu(), cos_packed, rtol=1e-5, atol=1e-6
    )


def test_default_position_ids_path_still_uses_cache():
    """position_ids=None keeps the cached arange behavior (and results)."""
    torch.manual_seed(9)
    cfg = small_config()
    batch = 2
    input_ids = torch.randint(0, cfg.vocab_size, (batch, 8))
    sin_ref, cos_ref = rope_sin_cos_from_position_ids(
        torch.arange(8).unsqueeze(0).expand(batch, 8),
        cfg.head_dim,
        rope_theta=cfg.rope_theta,
    )

    out_cached = run_case(cfg, 44, input_ids)
    out_explicit = run_case(cfg, 44, input_ids, sin=sin_ref, cos=cos_ref)
    assert torch.allclose(out_cached, out_explicit, rtol=1e-4, atol=1e-4)
