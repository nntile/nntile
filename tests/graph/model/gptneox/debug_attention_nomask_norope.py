#!/usr/bin/env python3
"""Stage-wise attention debug: HF vs NNTile reference (no RoPE, no mask).

Run from repo root after fixtures exist (ctest data setup or
``generate_test_data.py --write-attention-rope-mask-variants``).

Expectation after ``stbn`` SDPA fix in ``generate_test_data.py``: HF eager
attention and C++ ``GptneoxAttention`` both match the reference within ~1e-6
forward; the old ~1.5e-3 gap was ``tsbn`` vs ``stbn`` logits layout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention as PtAttention,
    apply_rotary_pos_emb,
)

# Import generators from sibling module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_test_data import (  # noqa: E402
    ATTENTION_DIMS,
    TestDims,
    _PtSdpaEagerFn,
    _apply_rope_hsbn,
    _gptneox_attn_forward,
    _gptneox_rope_dim,
    _hidden_hsbn,
    _make_config,
    _o_weight_fortran,
    _proj_o_bsh,
    _proj_qkv_hsbn,
    nntile_layout_to_logical,
)


def rel_frob(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    d = np.linalg.norm(a - b)
    s = max(np.linalg.norm(a), np.linalg.norm(b), 1e-30)
    return float(d / s)


def hsbn_to_bsh(t: torch.Tensor) -> torch.Tensor:
    """``(h,s,b)`` NNTile hidden -> HF ``(b,s,h)``."""
    return t.permute(2, 1, 0).contiguous()


def hsbn4_to_bhsd(t: torch.Tensor) -> torch.Tensor:
    """``(h,s,b,n)`` -> ``(b,n,s,h)``."""
    return t.permute(2, 3, 1, 0).contiguous()


def bhsd_to_hsbn4(t: torch.Tensor) -> torch.Tensor:
    return t.permute(3, 2, 0, 1).contiguous()


def hf_attention_no_mask_norope(
    attn: PtAttention,
    x_bsh: torch.Tensor,
    dims: TestDims,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """HF eager path: merged QKV, identity RoPE on rotary_ndims, no mask."""
    qkv = attn.query_key_value(x_bsh)
    qkv = qkv.view(
        x_bsh.shape[0],
        x_bsh.shape[1],
        attn.config.num_attention_heads,
        3 * attn.head_size,
    ).transpose(1, 2)
    q, k, v = qkv.split(attn.head_size, dim=-1)
    cos = torch.ones(
        x_bsh.shape[0],
        x_bsh.shape[1],
        attn.head_size,
        device=x_bsh.device,
        dtype=x_bsh.dtype,
    )
    sin = torch.zeros_like(cos)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    scale = attn.head_size ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    ctx = torch.matmul(probs, v)
    ctx = ctx.transpose(1, 2).contiguous().view(
        x_bsh.shape[0], x_bsh.shape[1], -1,
    )
    out = attn.dense(ctx)
    stages = {"q": q, "k": k, "v": v, "ctx_bhsd": ctx, "out_bsh": out}
    return out, stages


def nntile_ref_stages(
    data: dict[str, np.ndarray],
    dims: TestDims,
    prefix: str = "attn",
) -> dict[str, np.ndarray]:
    wq = data[f"{prefix}.q_weight"]
    wk = data[f"{prefix}.k_weight"]
    wv = data[f"{prefix}.v_weight"]
    wo = data[f"{prefix}.o_weight"]
    inp = nntile_layout_to_logical(data["input"])
    x_pt = torch.tensor(inp.T, dtype=torch.float32).requires_grad_(False)
    x_hsbn = _hidden_hsbn(x_pt)
    q, k, v = _proj_qkv_hsbn(wq, wk, wv, x_hsbn)
    cos = np.ones_like(np.asarray(data["rope_cos"]))
    sin = np.zeros_like(np.asarray(data["rope_sin"]))
    rope_dim = _gptneox_rope_dim(dims)
    q, k = _apply_rope_hsbn(q, k, cos, sin, rope_dim)
    ctx = _PtSdpaEagerFn.apply(q, k, v, None)
    out_hsbn = _proj_o_bsh(wo, ctx)
    return {
        "q": q.detach().numpy(),
        "k": k.detach().numpy(),
        "v": v.detach().numpy(),
        "sdpa": ctx.detach().numpy(),
        "out": out_hsbn.detach().numpy(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("build/tests/graph/model/gptneox_data"),
    )
    parser.add_argument("--stem", default="gptneox_attention_no_rope")
    args = parser.parse_args()

    st_path = args.data_dir / f"{args.stem}.safetensors"
    if not st_path.is_file():
        print(f"Missing {st_path}", file=sys.stderr)
        return 1

    data = load_file(str(st_path))
    dims = ATTENTION_DIMS
    config = _make_config(dims)
    attn = PtAttention(config, layer_idx=0)
    attn.eval()

    prefix = "attn"
    wq = data[f"{prefix}.q_weight"]
    wk = data[f"{prefix}.k_weight"]
    wv = data[f"{prefix}.v_weight"]
    wo = data[f"{prefix}.o_weight"]
    nh, hd, H = dims.n_heads, dims.head_size, dims.hidden
    qkv = np.zeros((3 * H, H), dtype=np.float32)
    for hi in range(nh):
        qkv[hi * hd : (hi + 1) * hd, :] = nntile_layout_to_logical(wq[hi])
        qkv[H + hi * hd : H + (hi + 1) * hd, :] = nntile_layout_to_logical(
            wk[hi],
        )
        qkv[2 * H + hi * hd : 2 * H + (hi + 1) * hd, :] = (
            nntile_layout_to_logical(wv[hi])
        )
    attn.query_key_value.weight.data = torch.tensor(qkv)
    o_logical = nntile_layout_to_logical(wo).reshape(H, H)
    attn.dense.weight.data = torch.tensor(o_logical)

    inp = nntile_layout_to_logical(data["input"])
    x_pt = torch.tensor(inp.transpose(2, 1, 0).copy(), dtype=torch.float32)
    x_bsh = x_pt

    ref_out_hsbn = nntile_layout_to_logical(data["output_ref"])
    # ``output_ref`` is Fortran (H,S,B); compare to (B,S,H) via transpose
    ref_out_bsh = ref_out_hsbn.transpose(2, 1, 0)
    cos_np = np.ones_like(np.asarray(data["rope_cos"], dtype=np.float32))
    sin_np = np.zeros_like(np.asarray(data["rope_sin"], dtype=np.float32))
    out_gen = _gptneox_attn_forward(
        attn, x_pt, cos_np, sin_np, dims, use_causal_mask=False,
    )
    out_gen_bsh = out_gen.detach().numpy()
    stages = nntile_ref_stages(data, dims)

    hf_out, hf_st = hf_attention_no_mask_norope(attn, x_bsh, dims)
    hf_out_np = hf_out.detach().numpy()

    print("=== Full output (no RoPE, no mask) ===")
    print(
        f"  regen _gptneox_attn_forward vs fixture: "
        f"{rel_frob(out_gen_bsh, ref_out_bsh):.6e}",
    )
    hf_vs_ref = rel_frob(hf_out_np, ref_out_bsh)
    hf_vs_regen = rel_frob(hf_out_np, out_gen_bsh)
    print(f"  HF vs fixture output_ref:         {hf_vs_ref:.6e}")
    print(f"  HF vs regen NNTile ref:           {hf_vs_regen:.6e}")

    print("\n=== Per-stage: HF vs NNTile ref ===")
    q_hf, k_hf, v_hf = hf_st["q"], hf_st["k"], hf_st["v"]
    q_nt = stages["q"]
    k_nt = stages["k"]
    v_nt = stages["v"]
    q_rel = rel_frob(bhsd_to_hsbn4(q_hf).detach().numpy(), q_nt)
    k_rel = rel_frob(bhsd_to_hsbn4(k_hf).detach().numpy(), k_nt)
    v_rel = rel_frob(bhsd_to_hsbn4(v_hf).detach().numpy(), v_nt)
    print(f"  Q  rel: {q_rel:.6e}")
    print(f"  K  rel: {k_rel:.6e}")
    print(f"  V  rel: {v_rel:.6e}")
    ctx_nt = stages["sdpa"]
    wo = _o_weight_fortran(attn, dims)
    ctx_nt_bsh = _proj_o_bsh(wo, torch.tensor(ctx_nt)).detach().numpy()
    ctx_nt_bsh = ctx_nt_bsh.transpose(2, 1, 0)
    # SDPA only
    ctx_hf_from_nt_qkv = _PtSdpaEagerFn.apply(
        bhsd_to_hsbn4(q_hf), bhsd_to_hsbn4(k_hf), bhsd_to_hsbn4(v_hf), None,
    )
    print(
        f"  SDPA (HF QKV -> nntile sdpa): "
        f"{rel_frob(ctx_hf_from_nt_qkv.detach().numpy(), ctx_nt):.6e}",
    )
    print("  SDPA (each native): compare via out_proj skipped")

    # HF QKV merged linear vs split gemm
    q_merged = torch.nn.functional.linear(x_bsh, attn.query_key_value.weight)
    q_merged = q_merged.view(
        dims.batch, dims.seq, dims.n_heads, 3 * dims.head_size,
    )[:, :, :, : dims.head_size].permute(0, 2, 1, 3)
    print(
        f"  Q HF linear vs HF-stage Q: "
        f"{rel_frob(q_merged.numpy(), q_hf.numpy()):.6e}",
    )
    print(
        f"  Q split-gemm vs NNTile ref Q: "
        f"{rel_frob(q_nt, q_nt):.6e} (sanity 0)",
    )
    q_split = _proj_qkv_hsbn(wq, wk, wv, _hidden_hsbn(x_bsh))[0]
    q_split_rel = rel_frob(
        bhsd_to_hsbn4(q_hf).detach().numpy(), q_split.numpy(),
    )
    print(f"  Q split-gemm vs HF Q: {q_split_rel:.6e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
