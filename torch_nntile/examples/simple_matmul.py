#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/simple_matmul.py
# Parallel distributed matrix multiplication with axis-group tiling.

"""Tiled matrix multiplication example using torch_nntile graph mode.

Records ``REPEAT`` matmuls, applies axis-group tiling, compiles and runs the
graph twice (two epochs on the same inputs). After both rounds, performs
follow-up PyTorch ops while the nntile session is still alive (this used to
trigger StarPU "handle is not initialized" on CUDA when tiling was enabled).

CPU example::

    export LD_LIBRARY_PATH=$PWD/build/nntile:/opt/starpu/lib
    export STARPU_SILENT=1 STARPU_FXT_TRACE=0 STARPU_WORKERS_NOBIND=1
    python torch_nntile/examples/simple_matmul.py \\
        --restrict-cpu --ncpu=$(nproc) --ncuda=0

CUDA example (manual, requires GPU)::

    export LD_LIBRARY_PATH=$PWD/build/nntile:/opt/starpu/lib
    python torch_nntile/examples/simple_matmul.py --restrict-cuda --ncuda=-1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "torch_nntile") not in sys.path:
    sys.path.insert(0, str(_REPO / "torch_nntile"))

import torch_nntile  # noqa: E402
from torch_nntile import _C  # noqa: E402

# Matrix shapes (smaller than the original 10k demo for faster CPU runs).
M = 4096
N = 3072
K = 4096
REPEAT = 10

# Real tiling: 2 tiles along M, N, and K.
MT = 2048
NT = 1536
KT = 2048


def apply_axis_tiling() -> None:
    """Re-apply tiling before each compile (cleared after compile_graph)."""
    torch_nntile.set_axis_group_tiling("M", MT)
    torch_nntile.set_axis_group_tiling("N", NT)
    torch_nntile.set_axis_group_tiling("K", KT)


def run_matmul_round(
    a_nnt: torch.Tensor,
    b_nnt: torch.Tensor,
    *,
    repeat: int,
    round_idx: int,
    print_groups: bool,
) -> tuple[torch.Tensor, float]:
    """Record matmuls, compile, run, and return the last output tensor."""
    for _ in range(repeat):
        c_nnt = a_nnt @ b_nnt

    torch_nntile.set_axis_group_name(a_nnt, {0: "M", 1: "K"})
    torch_nntile.set_axis_group_name(b_nnt, {1: "N"})

    apply_axis_tiling()
    if print_groups and round_idx == 1:
        torch_nntile.print_axis_groups()

    t0 = -time.time()
    torch_nntile.compile_graph()
    torch_nntile.run()
    torch_nntile.wait()
    elapsed = t0 + time.time()
    print(f"Round {round_idx}: matmul x{repeat} compile+run: {elapsed:.4f} s")
    return c_nnt, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=REPEAT)
    parser.add_argument("--ncpu", type=int, default=-1)
    parser.add_argument("--ncuda", type=int, default=-1)
    parser.add_argument(
        "--restrict-cpu",
        action="store_true",
        help="Pin nntile kernels to CPU workers",
    )
    parser.add_argument(
        "--restrict-cuda",
        action="store_true",
        help="Pin nntile kernels to CUDA workers (requires ncuda > 0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose StarPU / NNTile context logging",
    )
    parser.add_argument(
        "--print-axis-groups",
        action="store_true",
        help="Print axis groups after the first round",
    )
    args = parser.parse_args()

    if not _C.has_libnntile():
        raise SystemExit(
            "torch_nntile was built without libnntile. "
            "Set NNTILE_BUILD_DIR and rebuild."
        )

    torch_nntile.init_context(
        ncpu=args.ncpu,
        ncuda=args.ncuda,
        verbose=int(args.verbose),
        cpu_fallback=False,
        runtime_mode="graph",
    )
    if args.restrict_cuda:
        torch_nntile.restrict_cuda()
    elif args.restrict_cpu:
        torch_nntile.restrict_cpu()

    print(
        f"Shapes: {M}x{K} @ {K}x{N}, "
        f"tiling M={MT} N={NT} K={KT}, repeat={args.repeat}"
    )

    a = torch.randn(M, K)
    b = torch.randn(K, N)
    a_nnt = a.to(device="nntile")
    b_nnt = b.to(device="nntile")

    c_nnt, t1 = run_matmul_round(
        a_nnt,
        b_nnt,
        repeat=args.repeat,
        round_idx=1,
        print_groups=args.print_axis_groups,
    )
    c_round1 = c_nnt.cpu().clone()
    c_nnt, t2 = run_matmul_round(
        a_nnt,
        b_nnt,
        repeat=args.repeat,
        round_idx=2,
        print_groups=False,
    )

    use_cuda = args.restrict_cuda or (
        args.ncuda != 0 and not args.restrict_cpu
    )
    if use_cuda and torch.cuda.is_available():
        c_ref = a.cuda() @ b.cuda()
        c2 = c_nnt.cpu().cuda()
    else:
        c_ref = a @ b
        c2 = c_nnt.cpu()

    rel_err = torch.norm(c_ref - c2) / torch.norm(c_ref)
    rel_err_r1 = torch.norm(c_ref - c_round1) / torch.norm(c_ref)
    total_time = t1 + t2
    flops = 2e-12 * args.repeat * 2 * M * K * N / total_time
    print(
        f"Matmul {args.repeat} times x2 epochs {M}x{K} @ {K}x{N} -> {M}x{N}: "
        f"{total_time:.4f} s total"
    )
    print(f"Performance (both rounds): {flops:.4f} Tflops/s")
    print(f"Relative error vs PyTorch (round 1): {rel_err_r1.item():.6e}")
    print(f"Relative error vs PyTorch (round 2): {rel_err.item():.6e}")

    torch_nntile.shutdown_context()


if __name__ == "__main__":
    main()
