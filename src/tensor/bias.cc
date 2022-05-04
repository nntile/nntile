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
 * @date 2022-04-22
 * */

#include "nntile/tensor/bias.hh"
#include "nntile/tile/bias.hh"

namespace nntile
{

// Asynchronous bias
template<typename T>
void bias_async(const Tensor<T> &src, const Tensor<T> &dst, Index axis)
{
    // Check dimensions
    if(dst.ndim != src.ndim+1)
    {
        throw std::runtime_error("dst.ndim != src.ndim+1");
    }
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= dst.ndim)
    {
        throw std::runtime_error("axis >= dst.ndim");
    }
    // Check shapes of input tensors
    for(Index i = 0; i < axis; ++i)
    {
        if(dst.shape[i] != src.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i]");
        }
        if(dst.basetile_shape[i] != src.basetile_shape[i])
        {
            throw std::runtime_error("dst.basetile_shape[i] != "
                    "src.basetile_shape[i]");
        }
    }
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src.shape[i-1])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i-1]");
        }
        if(dst.basetile_shape[i] != src.basetile_shape[i-1])
        {
            throw std::runtime_error("dst.basetile_shape[i] != "
                    "src.basetile_shape[i-1]");
        }
    }
    // Now apply per-tile bias asynchronously as needed
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
            // Apply per-tile bias
            bias_async(src_tile, dst_tile, axis);
        }
    }
}

template
void bias_async(const Tensor<float> &src, const Tensor<float> &dst,
        Index axis=1);

template
void bias_async(const Tensor<double> &src, const Tensor<double> &dst,
        Index axis=1);

template<typename T>
void bias_avg_dev_async(const Tensor<T> &avg_dev, const Tensor<T> &dst,
        Index axis)
{
    // Check dimensions
    if(dst.ndim != avg_dev.ndim)
    {
        throw std::runtime_error("dst.ndim != avg_dev.ndim");
    }
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= dst.ndim)
    {
        throw std::runtime_error("axis >= dst.ndim");
    }
    // Check shapes of input tensors
    if(avg_dev.shape[0] != 2)
    {
        throw std::runtime_error("avg_dev.shape[0] != 2");
    }
    if(avg_dev.basetile_shape[0] != 2)
    {
        throw std::runtime_error("avg_dev.basetile_shape[0] != 2");
    }
    for(Index i = 0; i < axis; ++i)
    {
        if(dst.shape[i] != avg_dev.shape[i+1])
        {
            throw std::runtime_error("dst.shape[i] != avg_dev.shape[i+1]");
        }
        if(dst.basetile_shape[i] != avg_dev.basetile_shape[i+1])
        {
            throw std::runtime_error("dst.basetile_shape[i] != "
                    "avg_dev.basetile_shape[i+1]");
        }
    }
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != avg_dev.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != avg_dev.shape[i]");
        }
        if(dst.basetile_shape[i] != avg_dev.basetile_shape[i])
        {
            throw std::runtime_error("dst.basetile_shape[i] != "
                    "avg_dev.basetile_shape[i]");
        }
    }
    // Now apply per-tile bias asynchronously as needed
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
            bias_avg_dev_async(avg_dev_tile, dst_tile, axis);
        }
    }
}

template
void bias_avg_dev_async(const Tensor<fp32_t> &avg_dev,
        const Tensor<fp32_t> &dst, Index axis);

template
void bias_avg_dev_async(const Tensor<fp64_t> &avg_dev,
        const Tensor<fp64_t> &dst, Index axis);

} // namespace nntile

