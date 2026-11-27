# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# NNHaul is a task-based dynamic execution engine: one in-order CUDA
# stream executes submitted tasks while a reactive data manager moves
# bytes between the device and pinned host memory on demand.
#
# @file tests/torch_gemm_add_step.py
# Classic GEMM then add on an NNHaul build: records both ops, runs one step
# on CUDA worker 0, reads the result.
#
# @version 1.0.0

# Classic GEMM then the add that already works, on an NNHaul build.
# Records both ops, runs one step on CUDA worker 0, reads the result.
# torch.mm is the torch-native aten path and is not in this build.
# This uses torch_nntile.gemm (nntile::tensor::gemm).

import os

os.environ.setdefault("NNTILE_ENABLE_LOCAL_COMPILER", "1")

import torch

import torch_nntile
from torch_nntile.gemm import gemm


def main() -> None:
    if not torch_nntile.built_with_cuda():
        raise SystemExit("torch_nntile was not built with CUDA")
    if torch_nntile.is_context_initialized():
        raise SystemExit(
            "context already initialized; pass ncuda=1 before the first op"
        )
    # CUDA worker 0 is the default when ncuda > 0. Do not restrict to CPU.
    torch_nntile.init_context(ncpu=1, ncuda=1)
    with torch.no_grad():
        # [[1, 2, 3], [4, 5, 6]] @ [[7, 8], [9, 10], [11, 12]]
        # = [[58, 64], [139, 154]], then add [[1, 2], [3, 4]].
        lhs = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float32,
        ).to("nntile")
        rhs = torch.tensor(
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            dtype=torch.float32,
        ).to("nntile")
        bias = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
        ).to("nntile")
    prod = gemm(lhs, rhs, ndim=1, batch_ndim=0)
    out = prod + bias
    torch_nntile.compile_graph()
    torch_nntile.run()
    result = out.cpu()
    torch.testing.assert_close(
        result,
        torch.tensor(
            [[59.0, 66.0], [142.0, 158.0]],
            dtype=torch.float32,
        ),
    )
    print("nnhaul torch gemm add step ok", result.tolist())


if __name__ == "__main__":
    main()
