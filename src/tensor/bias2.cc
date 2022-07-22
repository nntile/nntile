/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/bias2.cc
 * Bias operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tensor/bias2.hh"
#include "nntile/tile/bias2.hh"

namespace nntile
{

// Normalization operation over single axis
template<typename T>
void bias2_work(const Tensor<T> &avg_dev, const Tensor<T> &dst,
        Index axis)
{
    // Apply per-tile bias asynchronously as needed
    for(Index i = 0; i < avg_dev.grid.nelems; ++i)
    {
        // Index of current source tile
        auto avg_dev_tile_index = avg_dev.grid.linear_to_index(i);
        // Source tile itself
        auto avg_dev_tile = avg_dev.get_tile(i);
        // Set fixed indices of current destination tile
        std::vector<Index> dst_tile_index(dst.ndim);
        for(Index j = 0; j < axis; ++j)
        {
            dst_tile_index[j] = avg_dev_tile_index[j+1];
        }
        for(Index j = axis+1; j < dst.ndim; ++j)
        {
            dst_tile_index[j] = avg_dev_tile_index[j];
        }
        // Loop through all necessary destination tiles
        for(Index j = 0; j < dst.grid.shape[axis]; ++j)
        {
            // Set floating axis
            dst_tile_index[axis] = j;
            // Get linear offset from index
            Index dst_tile_offset = dst.grid.index_to_linear(dst_tile_index);
            // Get destination tile
            auto dst_tile = dst.get_tile(dst_tile_offset);
            // Apply per-tile bias
            bias2_work(avg_dev_tile, dst_tile, axis);
        }
    }
}

// Explicit instantiation of template
template
void bias2_work(const Tensor<fp32_t> &avg_dev,
        const Tensor<fp32_t> &dst, Index axis);

// Explicit instantiation of template
template
void bias2_work(const Tensor<fp64_t> &avg_dev,
        const Tensor<fp64_t> &dst, Index axis);

} // namespace nntile

