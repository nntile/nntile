/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/copy.cc
 * Copy operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tensor/copy.hh"
#include "nntile/tile/copy.hh"

namespace nntile
{

template<typename T>
void copy_intersection_work(const Tile<T> &src,
        const std::vector<Index> &src_offset, const Tensor<T> &dst,
        const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch)
{
    Index ndim = src.ndim;
    // Treat special case of ndim=0
    if(ndim == 0)
    {
        copy_intersection_work(src, src_offset, dst.get_tile(0),
                dst_offset, scratch);
        return;
    }
    // Treat non-zero ndim
    for(Index j = 0; j < dst.grid.nelems; ++j)
    {
        auto dst_tile_index = dst.grid.linear_to_index(j);
        std::vector<Index> dst_tile_offset(dst_offset);
        for(Index k = 0; k < ndim; ++k)
        {
            dst_tile_offset[k] += dst_tile_index[k] *
                dst.basetile_shape[k];
        }
        copy_intersection_work<T>(src, src_offset,
                dst.get_tile(j), dst_tile_offset, scratch);
    }
}

template
void copy_intersection_work(const Tile<fp32_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp32_t> &dst,
        const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch);

template
void copy_intersection_work(const Tile<fp64_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp64_t> &dst,
        const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch);

template<typename T>
void copy_intersection_work(const Tensor<T> &src,
        const std::vector<Index> &src_offset, const Tile<T> &dst,
        const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch)
{
    Index ndim = src.ndim;
    // Treat special case of ndim=0
    if(ndim == 0)
    {
        copy_intersection_work(src.get_tile(0), src_offset, dst,
                dst_offset, scratch);
        return;
    }
    // Treat non-zero ndim
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        auto src_tile_index = src.grid.linear_to_index(i);
        std::vector<Index> src_tile_offset(src_offset);
        for(Index k = 0; k < ndim; ++k)
        {
            src_tile_offset[k] += src_tile_index[k] * src.basetile_shape[k];
        }
        auto src_tile = src.get_tile(i);
        copy_intersection_work<T>(src_tile, src_tile_offset,
                dst, dst_offset, scratch);
    }
}

template
void copy_intersection_work(const Tensor<fp32_t> &src,
        const std::vector<Index> &src_offset, const Tile<fp32_t> &dst,
        const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch);

template
void copy_intersection_work(const Tensor<fp64_t> &src,
        const std::vector<Index> &src_offset, const Tile<fp64_t> &dst,
        const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch);

template<typename T>
void copy_intersection_work(const Tensor<T> &src,
        const std::vector<Index> &src_offset, const Tensor<T> &dst,
        const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch)
{
    Index ndim = src.ndim;
    // Treat special case of ndim=0
    if(ndim == 0)
    {
        copy_intersection_work(src.get_tile(0), src_offset, dst.get_tile(0),
                dst_offset, scratch);
        return;
    }
    // Treat non-zero ndim
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        auto src_tile_index = src.grid.linear_to_index(i);
        std::vector<Index> src_tile_offset(src_offset);
        for(Index k = 0; k < ndim; ++k)
        {
            src_tile_offset[k] += src_tile_index[k] * src.basetile_shape[k];
        }
        auto src_tile = src.get_tile(i);
        for(Index j = 0; j < dst.grid.nelems; ++j)
        {
            auto dst_tile_index = dst.grid.linear_to_index(j);
            std::vector<Index> dst_tile_offset(dst_offset);
            for(Index k = 0; k < ndim; ++k)
            {
                dst_tile_offset[k] += dst_tile_index[k] *
                    dst.basetile_shape[k];
            }
            copy_intersection_work<T>(src_tile, src_tile_offset,
                    dst.get_tile(j), dst_tile_offset, scratch);
        }
    }
}

template
void copy_intersection_work(const Tensor<fp32_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp32_t> &dst,
        const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch);

template
void copy_intersection_work(const Tensor<fp64_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp64_t> &dst,
        const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch);

} // namespace nntile

