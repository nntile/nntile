#!/usr/bin/env python3
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file scripts/generate_llama_test_data.py
# Generate random Llama test data in NNTile safetensors format.
# Output is used by C++ tests to verify forward/backward via safetensors.
#
# Usage: python generate_llama_test_data.py [--output DIR] [--seed N]
#
# @version 1.1.0

"""
Generate random very small Llama building block test data.
Saves weights, input, expected output, and parameter gradients in safetensors.
Data layout matches NNTile C++ (Fortran/column-major).
Uses PyTorch transformers for reference output/gradients when available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from safetensors.numpy import save_file
except ImportError:
    print(
        "safetensors required: pip install safetensors",
        file=sys.stderr,
    )
    sys.exit(1)


def fortran_order(arr: np.ndarray) -> np.ndarray:
    """Ensure array is Fortran-order (column-major) for NNTile."""
    return np.asfortranarray(arr).astype(np.float32)


def pt_to_nntile_attn_weights(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, o: np.ndarray,
    n_heads: int, head_size: int, n_emb: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert PyTorch (out,in) 2D weights to NNTile 3D layout.
    PT q_proj: (n_emb, n_emb) row-major -> NNTile:
    (n_heads, head_size, n_emb) Fortran.
    """
    # PT weight (out, in); reshape out dim to (n_heads, head_size)
    q_3d = fortran_order(q.reshape(n_heads, head_size, n_emb).copy())
    k_3d = fortran_order(k.reshape(n_heads, head_size, n_emb).copy())
    v_3d = fortran_order(v.reshape(n_heads, head_size, n_emb).copy())
    # o_proj: (n_emb, n_emb) -> (n_emb, n_heads, head_size)
    o_3d = fortran_order(o.reshape(n_emb, n_heads, head_size).copy())
    return q_3d, k_3d, v_3d, o_3d


def generate_llama_attention_data(
    hidden: int = 8,
    n_heads: int = 1,
    head_size: int = 8,
    seq: int = 4,
    batch: int = 2,
    seed: int = 42,
    use_torch_ref: bool = True,
) -> dict[str, np.ndarray]:
    """Generate test data for LlamaAttention (non-GQA).
    Parameter names match C++ named_parameters_recursive for module 'attn'.
    """
    rng = np.random.default_rng(seed)
    n_emb = hidden

    if use_torch_ref:
        try:
            import torch
            from transformers import LlamaConfig
            from transformers.models.llama.modeling_llama import (
                LlamaAttention as PtLlamaAttention)

            torch.manual_seed(seed)
            config = LlamaConfig(
                hidden_size=n_emb,
                num_attention_heads=n_heads,
                num_key_value_heads=n_heads,
                intermediate_size=16,
                num_hidden_layers=1,
            )
            pt_attn = PtLlamaAttention(config, layer_idx=0)
            pt_attn.eval()

            # Get weights and convert to NNTile layout
            q = pt_attn.q_proj.weight.detach().numpy()
            k = pt_attn.k_proj.weight.detach().numpy()
            v = pt_attn.v_proj.weight.detach().numpy()
            o = pt_attn.o_proj.weight.detach().numpy()
            q_3d, k_3d, v_3d, o_3d = pt_to_nntile_attn_weights(
                q, k, v, o, n_heads, head_size, n_emb)

            # Input: (hidden, seq, batch) - NNTile layout
            x_np = rng.standard_normal((hidden, seq, batch)).astype(
                np.float32
            ) * 0.1
            x_pt = torch.from_numpy(np.ascontiguousarray(x_np.T))
            x_pt = x_pt.unsqueeze(0).permute(0, 2, 1, 3)
            x_pt.requires_grad_(True)

            out_pt, _, _ = pt_attn(
                x_pt, position_ids=None, past_key_value=None
            )
            out_pt = out_pt.squeeze(0).permute(2, 1, 0)  # (hidden, seq, batch)
            out_np = out_pt.detach().numpy()
            out_np = fortran_order(out_np)

            # Backward for gradients
            loss = out_pt.sum()
            loss.backward()
            q_grad = pt_attn.q_proj.weight.grad.numpy()
            k_grad = pt_attn.k_proj.weight.grad.numpy()
            v_grad = pt_attn.v_proj.weight.grad.numpy()
            o_grad = pt_attn.o_proj.weight.grad.numpy()
            qg, kg, vg, og = pt_to_nntile_attn_weights(
                q_grad, k_grad, v_grad, o_grad, n_heads, head_size, n_emb)

            input_hsb = fortran_order(x_np)
            return {
                "attn.q_weight": q_3d,
                "attn.k_weight": k_3d,
                "attn.v_weight": v_3d,
                "attn.o_weight": o_3d,
                "input": input_hsb,
                "output_ref": out_np,
                "attn.q_weight_grad": qg,
                "attn.k_weight_grad": kg,
                "attn.v_weight_grad": vg,
                "attn.o_weight_grad": og,
            }
        except ImportError:
            use_torch_ref = False

    # Fallback: random data without reference output
    def _rand(shape: tuple, scale: float) -> np.ndarray:
        return fortran_order(
            rng.standard_normal(shape).astype(np.float32) * scale
        )

    q_weight = _rand((n_heads, head_size, n_emb), 0.02)
    k_weight = _rand((n_heads, head_size, n_emb), 0.02)
    v_weight = _rand((n_heads, head_size, n_emb), 0.02)
    o_weight = _rand((n_emb, n_heads, head_size), 0.02)
    input_hsb = _rand((hidden, seq, batch), 0.1)
    output_hsb = _rand((hidden, seq, batch), 0.1)
    q_grad = _rand((n_heads, head_size, n_emb), 0.01)
    k_grad = _rand((n_heads, head_size, n_emb), 0.01)
    v_grad = _rand((n_heads, head_size, n_emb), 0.01)
    o_grad = _rand((n_emb, n_heads, head_size), 0.01)

    return {
        "attn.q_weight": q_weight,
        "attn.k_weight": k_weight,
        "attn.v_weight": v_weight,
        "attn.o_weight": o_weight,
        "input": input_hsb,
        "output_ref": output_hsb,
        "attn.q_weight_grad": q_grad,
        "attn.k_weight_grad": k_grad,
        "attn.v_weight_grad": v_grad,
        "attn.o_weight_grad": o_grad,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate Llama test data (safetensors)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output directory (default: tests/model/llama/data)",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed",
    )
    parser.add_argument(
        "--no-torch", action="store_true",
        help="Skip PyTorch reference (use random data only)",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    out_dir = args.output or str(base / "tests" / "model" / "llama" / "data")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    data = generate_llama_attention_data(
        seed=args.seed, use_torch_ref=not args.no_torch
    )
    weights_only = {
        k: v for k, v in data.items()
        if "grad" not in k and k not in ("input", "output_ref")
    }
    full_path = str(Path(out_dir) / "llama_attention.safetensors")
    save_file(weights_only, full_path)
    print(f"Saved weights to {full_path}")

    full_data_path = str(Path(out_dir) / "llama_attention_full.safetensors")
    save_file(data, full_data_path)
    print(f"Saved full test data to {full_data_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
