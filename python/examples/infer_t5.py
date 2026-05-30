# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file python/examples/infer_t5.py
# infer_t5.py
#
# @version 1.1.0

"""Inference t5 via NNTile graph API (scaffold)."""
from nntile import Context, NNGraph


def main() -> None:
    ctx = Context()
    _ = ctx, NNGraph()
    raise SystemExit('Example scaffold: implement t5 inference')


if __name__ == '__main__':
    main()
