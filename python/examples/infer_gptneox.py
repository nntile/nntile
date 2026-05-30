# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file python/examples/infer_gptneox.py
# infer_gptneox.py
#
# @version 1.1.0

"""Inference gptneox via NNTile graph API (scaffold)."""
from nntile import Context, NNGraph


def main() -> None:
    ctx = Context()
    _ = ctx, NNGraph()
    raise SystemExit('Example scaffold: implement gptneox inference')


if __name__ == '__main__':
    main()
