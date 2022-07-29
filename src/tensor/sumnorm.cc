/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/norm.hh
 * Functions that compute different norms.
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tensor/sumnorm.hh"
#include "nntile/tile/sumnorm.hh"
#include "nntile/tile/clear.hh"

namespace nntile
{

template<typename T>
void sumnorm_async(const Tensor<T> &src, const Tensor<T> &sumnorm, Index axis)
{
    // Check dimensions
    if(src.ndim != sumnorm.ndim)
    {
        throw std::runtime_error("src.ndim != sumnorm.ndim");
    }
    // Treat special case of src.ndim=0
    if(src.ndim == 0)
    {
        throw std::runtime_error("Scalar input makes no sense");
    }
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= src.ndim)
    {
        throw std::runtime_error("axis >= src.ndim");
    }
    // Check shapes of src and sumnorm
    if(sumnorm.shape[0] != 2)
    {
        throw std::runtime_error("sumnorm.shape[0] != 2");
    }
    if(sumnorm.basetile_shape[0] != 2)
    {
        throw std::runtime_error("sumnorm.basetile_shape[0] != 2");
    }
    for(Index i = 0; i < axis; ++i)
    {
        if(src.shape[i] != sumnorm.shape[i+1])
        {
            throw std::runtime_error("src.shape[i] != sumnorm.shape[i+1]");
        }
        if(src.basetile_shape[i] != sumnorm.basetile_shape[i+1])
        {
            throw std::runtime_error("src.basetile_shape[i] != "
                    "sumnorm.basetile_shape[i+1]");
        }
    }
    for(Index i = axis+1; i < src.ndim; ++i)
    {
        if(src.shape[i] != sumnorm.shape[i])
        {
            throw std::runtime_error("src.shape[i] != sumnorm.shape[i]");
        }
        if(src.basetile_shape[i] != sumnorm.basetile_shape[i])
        {
            throw std::runtime_error("src.basetile_shape[i] != "
                    "sumnorm.basetile_shape[i]");
        }
    }
    // Compute sum and sum of squares for each tile in grid of src tensor
    for(Index i = 0; i < sumnorm.grid.nelems; ++i)
    {
        auto dst_tile = sumnorm.get_tile(i);
        auto dst_tile_index = sumnorm.grid.linear_to_index(i);
        std::vector<Index> src_tile_index(src.ndim);
        for(Index j = 0; j < axis; ++j)
        {
            src_tile_index[j] = dst_tile_index[j+1];
        }
        src_tile_index[axis] = 0;
        for(Index j = axis+1; j < src.ndim; ++j)
        {
            src_tile_index[j] = dst_tile_index[j];
        }
        // Launch per-tile kernel
        for(Index j = 0; j < src.grid.shape[axis]; ++j)
        {
            src_tile_index[axis] = j;
            auto src_tile = src.get_tile(src_tile_index);
            sumnorm_async<T>(src_tile, dst_tile, axis);
        }
    }
}

// Explicit instantiation
template
void sumnorm_async(const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &sumnorm, Index axis);

template
void sumnorm_async(const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &sumnorm, Index axis);

} // namespace nntile

