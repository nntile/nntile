/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/distributions.hh
 * Distributions for tensors
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <vector>

namespace nntile::tensor::distributions
{

std::vector<int> block_cyclic(const std::vector<Index> &tensor_grid,
        const std::vector<int> &mpi_grid, int start_rank, int max_rank);

} // namespace nntile::tensor::distributions
