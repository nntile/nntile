# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/nntile_core/tensor/distributions.pyi
#
# @version 1.1.0

from typing import Sequence

def block_cyclic(tensor_grid: Sequence[int], mpi_grid: Sequence[int],
                 start_rank: int, max_rank: int) -> list[int]: ...
