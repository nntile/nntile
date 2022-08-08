/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/copy.cc
 * Copy operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-05
 * */

#include "nntile/tile/copy.hh"
#include "nntile/starpu/copy.hh"

namespace nntile
{

template<typename T>
void copy_work(const Tile<T> &src, const std::vector<Index> &src_start,
        const Tile<T> &dst, const std::vector<Index> &dst_start,
        const std::vector<Index> &copy_shape,
        const StarpuVariableHandle &scratch,
        enum starpu_data_access_mode mode)
{
    switch(mode)
    {
        case STARPU_W:
        case STARPU_RW:
        case Starpu::STARPU_RW_COMMUTE:
            break;
        default:
            throw std::runtime_error("Unsupported mode");
    }
    // Insert task
    nntile::starpu::copy<T>(src.ndim, src_start, src.stride, dst_start,
            dst.stride, copy_shape, src, dst, scratch, mode);
}

template<typename T>
void copy_work(const Tile<T> &src, const std::vector<Index> &src_offset,
        const Tile<T> &dst, const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch)
{
    Index ndim = src.ndim;
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
    // Launch codelet
    copy_work<T>(src, src_start, dst, dst_start, copy_shape,
            scratch, dst_tile_mode);
}

template
void copy_work(const Tile<fp32_t> &src, const std::vector<Index> &src_offset,
        const Tile<fp32_t> &dst, const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch);

template
void copy_work(const Tile<fp64_t> &src, const std::vector<Index> &src_offset,
        const Tile<fp64_t> &dst, const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch);

} // namespace nntile

