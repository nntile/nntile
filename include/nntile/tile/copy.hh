/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/copy.hh
 * Copy operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-04
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{

//! Asynchronous tile-wise copy operation
//
// @param[in] src: Source tile
// @param[in] src_offset: Initial offset of the source tile
// @param[inout] dst: Destination tile
// @param[in] dst_offset: Initial offset of the destination tile
//
// This operation finds an intersection of the source and the target tiles
// and copies only the data within the found intersection. No elements of the
// destination tile outside the intersection mask are updated.
template<typename T>
void copy_work(const Tile<T> &src,
        const std::vector<Index> &src_start, const Tile<T> &dst,
        const std::vector<Index> &dst_start,
        const std::vector<Index> &copy_shape,
        const StarpuVariableHandle &scratch,
        enum starpu_data_access_mode mode);

template<typename T>
void copy_work(const Tile<T> &src,
        const std::vector<Index> &src_offset, const Tile<T> &dst,
        const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch);

template<typename T>
void copy_work(const Tile<T> &src,
        const std::vector<Index> &src_offset, const Tile<T> &dst,
        const std::vector<Index> &dst_offset)
{
    // Treat special case of ndim=0
    if(src.ndim == 0)
    {
        starpu_data_cpy(dst, src, 1, nullptr, nullptr);
        return;
    }
    // Treat easy case of full copy
    if(src_offset == dst_offset and src.shape == dst.shape)
    {
        starpu_data_cpy(dst, src, 1, nullptr, nullptr);
        return;
    }
    // Do the slow partial copy
    // Temporary buffer for indexing
    StarpuVariableHandle scratch(2 * src.ndim * sizeof(Index));
    // Delegate computations
    copy_work<T>(src, src_offset, dst, dst_offset, scratch);
}

template<typename T>
void copy_async(const Tile<T> &src,
        const std::vector<Index> &src_offset, const Tile<T> &dst,
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
    // Delegate computations
    copy_work<T>(src, src_offset, dst, dst_offset);
}

//! Asynchronous tile-wise copy operation
//
// @param[in] src: Source tile
// @param[inout] dst: Destination tile
//
// This operation finds an intersection of the source and the target tiles
// and copies only the data within the found intersection. No elements of the
// destination tile outside the intersection mask are updated. Both the
// source and the target tiles assumed to have the same offset.
template<typename T>
void copy_async(const Tile<T> &src, const Tile<T> &dst)
{
    copy_async<T>(src, std::vector<Index>(src.ndim), dst,
            std::vector<Index>(dst.ndim));
}

//! Blocking version of tile-wise copy operation
//
// @param[in] src: Source tile
// @param[in] src_offset: Initial offset of the source tile
// @param[inout] dst: Destination tile
// @param[in] dst_offset: Initial offset of the destination tile
//
// This operation finds an intersection of the source and the target tiles
// and copies only the data within the found intersection. No elements of the
// destination tile outside the intersection mask are updated.
template<typename T>
void copy(const Tile<T> &src,
        const std::vector<Index> &src_offset, const Tile<T> &dst,
        const std::vector<Index> &dst_offset)
{
    copy_async<T>(src, src_offset, dst, dst_offset);
    starpu_task_wait_for_all();
}

//! Blocking version of tile-wise copy operation
//
// @param[in] src: Source tile
// @param[inout] dst: Destination tile
//
// This operation finds an intersection of the source and the target tiles
// and copies only the data within the found intersection. No elements of the
// destination tile outside the intersection mask are updated. Both the
// source and the target tiles assumed to have the same offset.
template<typename T>
void copy(const Tile<T> &src, const Tile<T> &dst)
{
    copy_async<T>(src, std::vector<Index>(src.ndim), dst,
            std::vector<Index>(dst.ndim));
    starpu_task_wait_for_all();
}

} // namespace nntile

