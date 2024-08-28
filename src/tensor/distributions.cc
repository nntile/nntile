/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/distributions.cc
 * Distributions for tensors
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/distributions.hh"
#include "nntile/tile/traits.hh"

namespace nntile::tensor::distributions
{

std::vector<int> block_cyclic(const std::vector<Index> &tensor_grid,
        const std::vector<int> &mpi_grid, int start_rank, int max_rank)
{
    // Check if dimensions of mpi grid and tensor match
    if(tensor_grid.size() != mpi_grid.size())
    {
        throw std::runtime_error("Wrong number of dimensions");
    }
    const Index ndim = mpi_grid.size();
    // Check starting and maximum ranks
    if(start_rank < 0 or start_rank >= max_rank)
    {
        throw std::runtime_error("Invalid starting rank");
    }
    // Define TileTraits object to use its linear_to_index method
    const tile::TileTraits traits(tensor_grid);
    // Define nodes/ranks for all tiles in a block-cyclic manner
    std::vector<int> ranks(traits.nelems, -1);
    for(Index i = 0; i < traits.nelems; ++i)
    {
        // Get index of a tile in the tensor
        auto index = traits.linear_to_index(i);
        // Get index within mpi grid/stamp
        for(Index j = 0; j < ndim; ++j)
        {
            index[j] %= mpi_grid[j];
        }
        // Obtain corresponding mpi rank
        int mpi_rank = index[ndim-1];
        for(Index j = ndim-2; j >= 0; --j)
        {
            mpi_rank = mpi_rank*mpi_grid[j] + index[j];
        }
        ranks[i] = (mpi_rank+start_rank) % max_rank;
    }
    return ranks;
}

} // namespace nntile::tensor::distributions
