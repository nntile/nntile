# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# NNHaul is a task-based dynamic execution engine: one in-order CUDA
# stream executes submitted tasks while a reactive data manager moves
# bytes between the device and pinned host memory on demand.
#
# @file tests/torch_add_step.py
# One classic PyTorch add on an NNHaul build: records a graph, runs one step
# on CUDA worker 0, reads the result.
#
# @version 1.0.0

# One classic PyTorch add on an NNHaul build.
# Records a graph, runs one step on CUDA worker 0, reads the result.

import os

os.environ.setdefault("NNTILE_ENABLE_LOCAL_COMPILER", "1")

import torch

import torch_nntile


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
        lhs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).to("nntile")
        rhs = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32).to("nntile")
    out = lhs + rhs
    torch_nntile.compile_graph()
    torch_nntile.run()
    result = out.cpu()
    torch.testing.assert_close(
        result,
        torch.tensor([5.0, 7.0, 9.0], dtype=torch.float32),
    )
    print("nnhaul torch add step ok", result.tolist())


if __name__ == "__main__":
    main()
