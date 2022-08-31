/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/distributions.hh
 * Distributions for tensors
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-29
 * */

#pragma once

#include <nntile/base_types.hh>
#include <vector>

namespace nntile
{
namespace tensor
{
namespace distributions
{

std::vector<int> block_cyclic(const std::vector<Index> &tensor_grid,
        const std::vector<int> &mpi_grid, int start_rank, int max_rank);

} // namespace distributions
} // namespace tensor
} // namespace nntile

