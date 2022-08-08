/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/bias.cc
 * Bias operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-08
 * */

#include "nntile/tensor/bias.hh"
#include "nntile/starpu/bias.hh"

namespace nntile
{

// Bias operation over single axis
template<typename T>
void bias_work(const Tensor<T> &src, const Tensor<T> &dst, Index axis)
{
    // Apply per-tile bias asynchronously as needed
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        // Index of current source tile
        auto src_tile_index = src.grid.linear_to_index(i);
        // Source tile itself
        auto src_tile = src.get_tile(i);
        // Set fixed indices of current destination tile
        std::vector<Index> dst_tile_index(dst.ndim);
        for(Index j = 0; j < axis; ++j)
        {
            dst_tile_index[j] = src_tile_index[j];
        }
        for(Index j = axis+1; j < dst.ndim; ++j)
        {
            dst_tile_index[j] = src_tile_index[j-1];
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
            // Reshape inputs: src_tile -> (m,n), dst_tile -> (m,k,n)
            Index m, n, k;
            if(axis == 0)
            {
                m = 1;
                n = src_tile.nelems;
                k = dst_tile.shape[0];
            }
            else if(axis == dst.ndim-1)
            {
                m = src_tile.nelems;
                n = 1;
                k = dst_tile.shape[axis];
            }
            else
            {
                m = dst_tile.stride[axis];
                n = dst_tile.matrix_shape[axis+1][1];
                k = dst_tile.shape[axis];
            }
            // Insert corresponding task
            nntile::starpu::bias<T>(m, n, k, src_tile, dst_tile);
        }
    }
}

// Explicit instantiation of template
template
void bias_work(const Tensor<fp32_t> &src, const Tensor<fp32_t> &dst,
        Index axis);

// Explicit instantiation of template
template
void bias_work(const Tensor<fp64_t> &src, const Tensor<fp64_t> &dst,
        Index axis);

} // namespace nntile

