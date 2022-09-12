/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/copy_intersection.cc
 * Copy intersection of 2 tiles from one into another
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-12
 * */

#include "nntile/tile/copy_intersection.hh"
#include "nntile/starpu/copy_intersection.hh"

namespace nntile
{
namespace tile
{

//! Asynchronous version of tile-wise copy operation
/*! This operation finds an intersection of the source and the target tiles
 * and copies only the data within the found intersection. No elements of the
 * destination tile outside the intersection mask are updated.
 *
 * @param[in] src: Source tile
 * @param[in] src_offset: Initial offset of the source tile
 * @param[inout] dst: Destination tile
 * @param[in] dst_offset: Initial offset of the destination tile
 * */
template<typename T>
void copy_intersection_async(const Tile<T> &src,
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
    Index ndim = src.ndim;
    // Treat special case of ndim=0
    if(ndim == 0)
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
    StarpuVariableHandle scratch(2 * ndim * sizeof(Index));
    // Perform smart copy
    std::vector<Index> src_start(ndim), dst_start(ndim), copy_shape(ndim);
    enum starpu_data_access_mode dst_tile_mode = STARPU_W;
    // Obtain starting indices and shape of intersection for copying
    for(Index i = 0; i < ndim; ++i)
    {
        // Do nothing if tiles do not intersect
        if((src_offset[i]+src.shape[i] <= dst_offset[i])
                or (dst_offset[i]+dst.shape[i] <= src_offset[i]))
        {
            return;
        }
        // Copy to the beginning of destination
        if(src_offset[i] < dst_offset[i])
        {
            dst_start[i] = 0;
            src_start[i] = dst_offset[i] - src_offset[i];
            copy_shape[i] = std::min(src.shape[i]-src_start[i],
                    dst.shape[i]);
        }
        // Copy from the beginning of source
        else
        {
            dst_start[i] = src_offset[i] - dst_offset[i];
            src_start[i] = 0;
            copy_shape[i] = std::min(dst.shape[i]-dst_start[i],
                    src.shape[i]);
        }
        // Check if destination is fully inside source
        if(copy_shape[i] != dst.shape[i])
        {
            dst_tile_mode = STARPU_RW;
        }
    }
    // Insert task
    starpu::copy_intersection::submit<T>(src.ndim, src_start, src.stride,
            dst_start, dst.stride, copy_shape, src, dst, scratch,
            dst_tile_mode);
}

//! Blocking version of tile-wise copy operation
/*! This operation finds an intersection of the source and the target tiles
 * and copies only the data within the found intersection. No elements of the
 * destination tile outside the intersection mask are updated.
 *
 * @param[in] src: Source tile
 * @param[in] src_offset: Initial offset of the source tile
 * @param[inout] dst: Destination tile
 * @param[in] dst_offset: Initial offset of the destination tile
 * */
template<typename T>
void copy_intersection(const Tile<T> &src,
        const std::vector<Index> &src_offset, const Tile<T> &dst,
        const std::vector<Index> &dst_offset)
{
    copy_intersection_async<T>(src, src_offset, dst, dst_offset);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void copy_intersection<fp32_t>(const Tile<fp32_t> &src,
        const std::vector<Index> &src_offset, const Tile<fp32_t> &dst,
        const std::vector<Index> &dst_offset);

template
void copy_intersection<fp64_t>(const Tile<fp64_t> &src,
        const std::vector<Index> &src_offset, const Tile<fp64_t> &dst,
        const std::vector<Index> &dst_offset);

} // namespace tile
} // namespace nntile
