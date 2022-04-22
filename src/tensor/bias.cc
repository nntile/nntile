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

template<typename T>
void bias_async(const Tensor<T> &src, const Tensor<T> &dst, Index batch_dim)
{
    if(dst.ndim != src.ndim+1)
    {
        throw std::runtime_error("dst.ndim != src.ndim+1");
    }
    if(batch_dim < 0)
    {
        throw std::runtime_error("batch_dim < 0");
    }
    for(Index i = 0; i < batch_dim; ++i)
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
    for(Index i = batch_dim+1; i < dst.ndim; ++i)
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
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        auto src_tile_index = src.grid.linear_to_index(i);
        auto src_tile = src.get_tile(i);
        std::vector<Index> dst_tile_index(dst.ndim);
        for(Index j = 0; j < batch_dim; ++j)
        {
            dst_tile_index[j] = src_tile_index[j];
        }
        for(Index j = batch_dim+1; j < dst.ndim; ++j)
        {
            dst_tile_index[j] = src_tile_index[j-1];
        }
        for(Index j = 0; j < dst.grid.shape[batch_dim]; ++j)
        {
            dst_tile_index[batch_dim] = j;
            Index dst_tile_offset = dst.grid.index_to_linear(dst_tile_index);
            auto dst_tile = dst.get_tile(dst_tile_offset);
            bias_async(src_tile, dst_tile, batch_dim);
        }
    }
}

template
void bias_async(const Tensor<float> &src, const Tensor<float> &dst,
        Index batch_dim=1);

template
void bias_async(const Tensor<double> &src, const Tensor<double> &dst,
        Index batch_dim=1);

} // namespace nntile

