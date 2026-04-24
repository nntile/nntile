/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/copy_intersection.cc
 * Copy intersection of 2 tensors from one into another
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/copy_intersection.hh"
#include "nntile/tile/copy_intersection.hh"
#include "nntile/starpu/config.hh"

#include <algorithm>

namespace nntile::tensor
{

//! Asynchronous tensor-wise copy operation
/*! This operation finds an intersection of the source and the target tensors
 * and copies only the data within the found intersection. No elements of the
 * destination tensor outside the intersection mask are updated. Both the
 * source and the target tensors assumed to have the same offset.
 *
 * @param[in] src: Source tensor
 * @param[in] src_offset: Initial offset of the source tensor
 * @param[inout] dst: Destination tensor
 * @param[in] dst_offset: Initial offset of the destination tensor
 * */
template<typename T>
void copy_intersection_async(const Tensor<T> &src,
        const std::vector<Index> &src_offset, const Tensor<T> &dst,
        const std::vector<Index> &dst_offset)
{
    // Check dimensions
    if(src.ndim != src_offset.size())
    {
        throw std::runtime_error("src.ndim != src_offset.size()");
    }
    if(src.ndim != dst.ndim)
    {
        throw std::runtime_error("src.ndim != dst.ndim");
    }
    if(dst.ndim != dst_offset.size())
    {
        throw std::runtime_error("dst.ndim != dst_offset.size()");
    }
    Index ndim = src.ndim;
    // Treat special case of ndim=0
    if(ndim == 0)
    {
        // TileTraits require positive extents; ndim==0 copy ignores scratch size.
        tile::Tile<int64_t> scratch({1});
        tile::copy_intersection_async<T>(src.get_tile(0), src_offset, dst.get_tile(0),
            dst_offset, scratch);
        dst.get_tile(0).mpi_flush();
        scratch.unregister_submit();
        return;
    }
    // Treat easy case of full copy
    if(src_offset == dst_offset and src.shape == dst.shape
            and src.basetile_shape == dst.basetile_shape)
    {
        tile::Tile<int64_t> scratch({2 * ndim});
        for(Index i = 0; i < src.grid.nelems; ++i)
        {
            std::vector<Index> tile_ix = src.grid.linear_to_index(i);
            std::vector<Index> corner(ndim);
            for(Index j = 0; j < ndim; ++j)
            {
                corner[j] = tile_ix[j] * src.basetile_shape[j];
            }
            tile::copy_intersection_async<T>(src.get_tile(i), corner, dst.get_tile(i),
                corner, scratch);
            dst.get_tile(i).mpi_flush();
        }
        scratch.unregister_submit();
        return;
    }
    // Do the slow complex copy
    tile::Tile<int64_t> scratch({2 * ndim});
    std::vector<Index> src_start(ndim), dst_start(ndim), copy_shape(ndim);
    std::vector<Index> dst_tile_index_begin(ndim), dst_tile_index_end(ndim);
    Index dst_ntiles = 1;
    for(Index i = 0; i < ndim; ++i)
    {
        if((src_offset[i]+src.shape[i] <= dst_offset[i])
                or (dst_offset[i]+dst.shape[i] <= src_offset[i]))
        {
            scratch.unregister_submit();
            return;
        }
        if(src_offset[i] < dst_offset[i])
        {
            src_start[i] = dst_offset[i] - src_offset[i];
            dst_start[i] = 0;
            copy_shape[i] = std::min(src.shape[i]-src_start[i],
                    dst.shape[i]);
        }
        else
        {
            src_start[i] = 0;
            dst_start[i] = src_offset[i] - dst_offset[i];
            copy_shape[i] = std::min(dst.shape[i]-dst_start[i],
                    src.shape[i]);
        }
        dst_tile_index_begin[i] = dst_start[i] / dst.basetile_shape[i];
        dst_tile_index_end[i] = (dst_start[i]+copy_shape[i]-1)
            / dst.basetile_shape[i] + 1;
        dst_ntiles *= dst_tile_index_end[i] - dst_tile_index_begin[i];
    }
    std::vector<Index> dst_tile_index(dst_tile_index_begin);
    for(Index i = 0; i < dst_ntiles; ++i)
    {
        Index dst_tile_offset = dst.grid.index_to_linear(dst_tile_index);
        auto dst_tile_handle = dst.get_tile(dst_tile_offset);
        Index src_ntiles = 1;
        std::vector<Index> src_tile_index_begin(ndim),
            src_tile_index_end(ndim);
        for(Index j = 0; j < ndim; ++j)
        {
            if(dst_tile_index[j] == dst_tile_index_begin[j])
            {
                src_tile_index_begin[j] = src_start[j] / src.basetile_shape[j];
            }
            else
            {
                src_tile_index_begin[j] =
                    (dst_tile_index[j]*dst.basetile_shape[j]
                     -dst_start[j]+src_start[j]) / src.basetile_shape[j];
            }
            if(dst_tile_index[j]+1 == dst_tile_index_end[j])
            {
                src_tile_index_end[j] = (src_start[j]+copy_shape[j]-1)
                    /src.basetile_shape[j] + 1;
            }
            else
            {
                src_tile_index_end[j] =
                    ((dst_tile_index[j]+1)*dst.basetile_shape[j]-1
                     -dst_start[j]+src_start[j]) / src.basetile_shape[j] + 1;
            }
            src_ntiles *= src_tile_index_end[j] - src_tile_index_begin[j];
        }
        std::vector<Index> src_tile_index = src_tile_index_begin;
        for(Index j = 0; j < src_ntiles; ++j)
        {
            Index src_tile_offset = src.grid.index_to_linear(src_tile_index);
            std::vector<Index> src_corner(ndim), dst_corner(ndim);
            for(Index k = 0; k < ndim; ++k)
            {
                src_corner[k] = src_tile_index[k] * src.basetile_shape[k];
                dst_corner[k] = dst_tile_index[k] * dst.basetile_shape[k]
                    - dst_start[k] + src_start[k];
            }
            tile::copy_intersection_async<T>(src.get_tile(src_tile_offset), src_corner,
                dst_tile_handle, dst_corner, scratch);
            if(j + 1 < src_ntiles)
            {
                ++src_tile_index[0];
                Index k = 0;
                while(src_tile_index[k] == src_tile_index_end[k])
                {
                    src_tile_index[k] = src_tile_index_begin[k];
                    ++k;
                    ++src_tile_index[k];
                }
            }
        }
        dst_tile_handle.mpi_flush();
        if(i == dst_ntiles-1)
        {
            break;
        }
        ++dst_tile_index[0];
        Index k = 0;
        while(dst_tile_index[k] == dst_tile_index_end[k])
        {
            dst_tile_index[k] = dst_tile_index_begin[k];
            ++k;
            ++dst_tile_index[k];
        }
    }
    scratch.unregister_submit();
}

//! Blocking version of tensor-wise copy operation
/*! This operation finds an intersection of the source and the target tensors
 * and copies only the data within the found intersection. No elements of the
 * destination tensor outside the intersection mask are updated.
 *
 * @param[in] src: Source tensor
 * @param[in] src_offset: Initial offset of the source tensor
 * @param[inout] dst: Destination tensor
 * @param[in] dst_offset: Initial offset of the destination tensor
 * */
template<typename T>
void copy_intersection(const Tensor<T> &src,
        const std::vector<Index> &src_offset, const Tensor<T> &dst,
        const std::vector<Index> &dst_offset)
{
    copy_intersection_async<T>(src, src_offset, dst, dst_offset);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void copy_intersection_async<bool_t>(const Tensor<bool_t> &src,
        const std::vector<Index> &src_offset, const Tensor<bool_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection_async<fp32_t>(const Tensor<fp32_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp32_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection_async<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp32_fast_tf32_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection_async<fp32_fast_fp16_t>(const Tensor<fp32_fast_fp16_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp32_fast_fp16_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection_async<fp32_fast_bf16_t>(const Tensor<fp32_fast_bf16_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp32_fast_bf16_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection_async<fp64_t>(const Tensor<fp64_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp64_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection_async<int64_t>(const Tensor<int64_t> &src,
        const std::vector<Index> &src_offset, const Tensor<int64_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection_async<fp16_t>(const Tensor<fp16_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp16_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection_async<bf16_t>(const Tensor<bf16_t> &src,
        const std::vector<Index> &src_offset, const Tensor<bf16_t> &dst,
        const std::vector<Index> &dst_offset);

// Explicit instantiation
template
void copy_intersection<bool_t>(const Tensor<bool_t> &src,
        const std::vector<Index> &src_offset, const Tensor<bool_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection<fp32_t>(const Tensor<fp32_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp32_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp32_fast_tf32_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection<fp32_fast_fp16_t>(const Tensor<fp32_fast_fp16_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp32_fast_fp16_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection<fp32_fast_bf16_t>(const Tensor<fp32_fast_bf16_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp32_fast_bf16_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection<fp64_t>(const Tensor<fp64_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp64_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection<int64_t>(const Tensor<int64_t> &src,
        const std::vector<Index> &src_offset, const Tensor<int64_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection<fp16_t>(const Tensor<fp16_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp16_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection<bf16_t>(const Tensor<bf16_t> &src,
        const std::vector<Index> &src_offset, const Tensor<bf16_t> &dst,
        const std::vector<Index> &dst_offset);

} // namespace nntile::tensor
