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
        --size small --restrict-cpu --ncpu=$(nproc) --ncuda=0

CUDA example (manual, requires GPU)::

    export LD_LIBRARY_PATH=$PWD/build/nntile:/opt/starpu/lib
    python torch_nntile/examples/simple_matmul.py \\
        --size small --restrict-cuda --ncuda=-1
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "torch_nntile") not in sys.path:
    sys.path.insert(0, str(_REPO / "torch_nntile"))

import torch_nntile  # noqa: E402


@dataclass(frozen=True)
class MatmulBenchmarkSize:
    """Square GEMM shapes and axis-group tile sizes."""

    m: int
    n: int
    k: int
    mt: int
    nt: int
    kt: int
    repeat: int


SIZE_PRESETS: dict[str, MatmulBenchmarkSize] = {
    "tiny": MatmulBenchmarkSize(
        m=1024,
        n=1024,
        k=1024,
        mt=384,
        nt=384,
        kt=256,
        repeat=10,
    ),
    "small": MatmulBenchmarkSize(
        m=4096,
        n=4096,
        k=4096,
        mt=1536,
        nt=1536,
        kt=1024,
        repeat=20,
    ),
    "medium": MatmulBenchmarkSize(
        m=6144,
        n=6144,
        k=6144,
        mt=2048,
        nt=2048,
        kt=2048,
        repeat=30,
    ),
    "large": MatmulBenchmarkSize(
        m=10240,
        n=10240,
        k=10240,
        mt=4096,
        nt=3072,
        kt=3072,
        repeat=50,
    ),
}


def apply_axis_tiling(mt: int, nt: int, kt: int) -> None:
    """Re-apply tiling before each compile (cleared after compile_graph)."""
    torch_nntile.set_axis_group_tiling("M", mt)
    torch_nntile.set_axis_group_tiling("N", nt)
    torch_nntile.set_axis_group_tiling("K", kt)


def run_matmul_round(
    a_nnt: torch.Tensor,
    b_nnt: torch.Tensor,
    *,
    size: MatmulBenchmarkSize,
    repeat: int,
    round_idx: int,
    print_groups: bool,
) -> tuple[torch.Tensor, float]:
    """Record matmuls, compile, run, and return the last output tensor."""
    for _ in range(repeat):
        c_nnt = a_nnt @ b_nnt

    torch_nntile.set_axis_group_name(a_nnt, {0: "M", 1: "K"})
    torch_nntile.set_axis_group_name(b_nnt, {1: "N"})

    apply_axis_tiling(size.mt, size.nt, size.kt)
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
    parser.add_argument(
        "--size",
        choices=tuple(SIZE_PRESETS),
        default="small",
        help="Benchmark preset (default: small)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="Override matmul count per epoch (default: preset value)",
    )
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

    size = SIZE_PRESETS[args.size]
    repeat = size.repeat if args.repeat is None else args.repeat

    torch_nntile.init_context(
        ncpu=args.ncpu,
        ncuda=args.ncuda,
        verbose=int(args.verbose),
        cpu_fallback=False,
    )
    if args.restrict_cuda:
        torch_nntile.restrict_cuda()
    elif args.restrict_cpu:
        torch_nntile.restrict_cpu()

    print(
        f"Size preset: {args.size} | "
        f"Shapes: {size.m}x{size.k} @ {size.k}x{size.n}, "
        f"tiling M={size.mt} N={size.nt} K={size.kt}, repeat={repeat}"
    )

    a = torch.randn(size.m, size.k)
    b = torch.randn(size.k, size.n)
    with torch.no_grad():
        a_nnt = a.to(device="nntile")
        b_nnt = b.to(device="nntile")

    c_nnt, t1 = run_matmul_round(
        a_nnt,
        b_nnt,
        size=size,
        repeat=repeat,
        round_idx=1,
        print_groups=args.print_axis_groups,
    )
    with torch.no_grad():
        c_round1 = c_nnt.cpu().clone()
    c_nnt, t2 = run_matmul_round(
        a_nnt,
        b_nnt,
        size=size,
        repeat=repeat,
        round_idx=2,
        print_groups=False,
    )

    use_cuda = args.restrict_cuda or (
        args.ncuda != 0 and not args.restrict_cpu
    )
    with torch.no_grad():
        if use_cuda and torch.cuda.is_available():
            c_ref = a.cuda() @ b.cuda()
            c2 = c_nnt.cpu().cuda()
            c_round1 = c_round1.cuda()
        else:
            c_ref = a @ b
            c2 = c_nnt.cpu()

        rel_err = torch.norm(c_ref - c2) / torch.norm(c_ref)
        rel_err_r1 = torch.norm(c_ref - c_round1) / torch.norm(c_ref)
    total_time = t1 + t2
    flops = 2e-12 * repeat * 2 * size.m * size.k * size.n / total_time
    print(
        f"Matmul {repeat} times x2 epochs "
        f"{size.m}x{size.k} @ {size.k}x{size.n} -> {size.m}x{size.n}: "
        f"{total_time:.4f} s total"
    )
    print(f"Performance (both rounds): {flops:.4f} Tflops/s")
    print(f"Relative error vs PyTorch (round 1): {rel_err_r1.item():.6e}")
    print(f"Relative error vs PyTorch (round 2): {rel_err.item():.6e}")

    torch_nntile.shutdown_context()


if __name__ == "__main__":
    main()
