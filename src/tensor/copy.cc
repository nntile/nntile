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
void copy_work(const Tile<T> &src,
        const std::vector<Index> &src_offset, const Tensor<T> &dst,
        const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch)
{
    Index ndim = src.ndim;
    std::vector<Index> src_start(ndim), dst_start(ndim), copy_shape(ndim);
    std::vector<Index> dst_tile_index_begin(ndim), dst_tile_index_end(ndim);
    Index dst_ntiles = 1;
    // Obtain starting indices and shape of intersection for copying
    for(Index i = 0; i < ndim; ++i)
    {
        // Do nothing if tensors do not intersect
        if((src_offset[i]+src.shape[i] <= dst_offset[i])
                or (dst_offset[i]+dst.shape[i] <= src_offset[i]))
        {
            return;
        }
        // Copy to the beginning of destination
        if(src_offset[i] < dst_offset[i])
        {
            src_start[i] = dst_offset[i] - src_offset[i]; // positive value
            dst_start[i] = 0;
            copy_shape[i] = std::min(src.shape[i]-src_start[i],
                    dst.shape[i]);
        }
        // Copy from the beginning of source
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
    // If there is only one destination tile
    if(dst_ntiles == 1)
    {
        auto dst_tile = dst.get_tile(dst_tile_index_begin);
        std::vector<Index> dst_tile_start(ndim);
        enum starpu_data_access_mode dst_tile_mode = STARPU_W;
        // Check if destination tile is only partially overwritten
        for(Index k = 0; k < ndim; ++k)
        {
            dst_tile_start[k] = dst_start[k]
                - dst_tile_index_begin[k]*dst.basetile_shape[k];
            if(dst_tile_start[k] != 0 or copy_shape[k] != dst_tile.shape[k])
            {
                dst_tile_mode = STARPU_RW;
            }
        }
        // Check if we can simply copy entire buffer
        if(dst_tile_mode == STARPU_W and copy_shape == src.shape)
        {
            starpu_data_cpy(dst_tile, src, 1, nullptr, nullptr);
        }
        // Smart copying otherwise
        else
        {
            copy_work(src, src_start, dst_tile, dst_tile_start,
                    copy_shape, scratch, dst_tile_mode);
        }
    }
    // Cycle through all destination tiles
    else
    {
        std::vector<Index> dst_tile_index(dst_tile_index_begin);
        for(Index i = 0; i < dst_ntiles; ++i)
        {
            auto dst_tile = dst.get_tile(dst_tile_index);
            enum starpu_data_access_mode dst_tile_mode = STARPU_W;
            // Check if destination tile is only partially overwritten
            for(Index j = 0; j < ndim; ++j)
            {
                if(dst_tile_index[j] == dst_tile_index_begin[j])
                {
                    if(dst_tile_index[j]*dst.basetile_shape[j] != dst_start[j])
                    {
                        dst_tile_mode = STARPU_RW;
                    }
                }
                if(dst_tile_index[j]+1 == dst_tile_index_end[j])
                {
                    if(dst_tile_index[j]*dst.basetile_shape[j]+dst_tile.shape[j]
                            != dst_start[j]+copy_shape[j])
                    {
                        dst_tile_mode = STARPU_RW;
                    }
                }
            }
            // Get starting indices of source and destination tiles and deduce
            // shape of copy
            std::vector<Index> src_tile_start(ndim), dst_tile_start(ndim),
                copy_tile_shape(ndim);
            for(Index k = 0; k < ndim; ++k)
            {
                if(dst_tile_index[k] == dst_tile_index_begin[k])
                {
                    src_tile_start[k] = src_start[k];
                    dst_tile_start[k] = dst_start[k]
                        - dst_tile_index[k]*dst.basetile_shape[k];
                }
                else
                {
                    src_tile_start[k] = src_start[k] - dst_start[k]
                        + dst_tile_index[k]*dst.basetile_shape[k];
                    dst_tile_start[k] = 0;
                }
                if(dst_tile_index[k]+1 == dst_tile_index_end[k])
                {
                    copy_tile_shape[k] = src_start[k] + copy_shape[k]
                        - src_tile_start[k];
                }
                else
                {
                    copy_tile_shape[k] = dst.basetile_shape[k]
                        - dst_tile_start[k];
                }
            }
            // Smart copy
            copy_work(src, src_tile_start, dst_tile,
                    dst_tile_start, copy_tile_shape, scratch,
                    dst_tile_mode);
            // Get out if it was the last tile
            if(i == dst_ntiles-1)
            {
                break;
            }
            // Get next tile
            ++dst_tile_index[0];
            Index k = 0;
            while(dst_tile_index[k] == dst_tile_index_end[k])
            {
                dst_tile_index[k] = dst_tile_index_begin[k];
                ++k;
                ++dst_tile_index[k];
            }
        }
    }
}

template
void copy_work(const Tile<fp32_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp32_t> &dst,
        const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch);

template
void copy_work(const Tile<fp64_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp64_t> &dst,
        const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch);

template<typename T>
void copy_work(const Tensor<T> &src,
        const std::vector<Index> &src_offset, const Tile<T> &dst,
        const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch)
{
    Index ndim = src.ndim;
    std::vector<Index> src_start(ndim), dst_start(ndim), copy_shape(ndim);
    std::vector<Index> src_tile_index_begin(ndim), src_tile_index_end(ndim);
    Index src_ntiles = 1;
    enum starpu_data_access_mode dst_mode = STARPU_W;
    // Obtain starting indices and shape of intersection for copying
    for(Index i = 0; i < ndim; ++i)
    {
        // Do nothing if tensors do not intersect
        if((src_offset[i]+src.shape[i] <= dst_offset[i])
                or (dst_offset[i]+dst.shape[i] <= src_offset[i]))
        {
            return;
        }
        // Copy to the beginning of destination
        if(src_offset[i] < dst_offset[i])
        {
            src_start[i] = dst_offset[i] - src_offset[i]; // positive value
            dst_start[i] = 0;
            copy_shape[i] = std::min(src.shape[i]-src_start[i],
                    dst.shape[i]);
        }
        // Copy from the beginning of source
        else
        {
            src_start[i] = 0;
            dst_start[i] = src_offset[i] - dst_offset[i];
            copy_shape[i] = std::min(dst.shape[i]-dst_start[i],
                    src.shape[i]);
        }
        // Check if destination will be fully overwritten or not
        if(dst_start[i] != 0 or copy_shape[i] != dst.shape[i])
        {
            dst_mode = STARPU_RW;
        }
        src_tile_index_begin[i] = src_start[i] / src.basetile_shape[i];
        src_tile_index_end[i] = (src_start[i]+copy_shape[i]-1)
            / src.basetile_shape[i] + 1;
        src_ntiles *= src_tile_index_end[i] - src_tile_index_begin[i];
    }
    // If there is only one corresponding source tile
    if(src_ntiles == 1)
    {
        auto src_tile = src.get_tile(src_tile_index_begin);
        // Check if we can simply copy entire buffer
        if(dst_mode == STARPU_W and copy_shape == src_tile.shape)
        {
            starpu_data_cpy(dst, src_tile, 1, nullptr, nullptr);
        }
        // Smart copying otherwise
        else
        {
            std::vector<Index> src_tile_start(ndim);
            for(Index k = 0; k < ndim; ++k)
            {
                src_tile_start[k] = src_start[k]
                    - src_tile_index_begin[k]*src.basetile_shape[k];
            }
            copy_work(src_tile, src_tile_start, dst,
                    dst_start, copy_shape, scratch, dst_mode);
        }
    }
    else
    {
        // Process the first source tile separately
        auto src_tile = src.get_tile(src_tile_index_begin);
        std::vector<Index> src_tile_start(ndim), copy_tile_shape(ndim);
        for(Index k = 0; k < ndim; ++k)
        {
            src_tile_start[k] = src_start[k]
                - src_tile_index_begin[k]*src.basetile_shape[k];
            if(src_tile_index_begin[k]+1 == src_tile_index_end[k])
            {
                copy_tile_shape[k] = copy_shape[k];
            }
            else
            {
                copy_tile_shape[k] = src.basetile_shape[k]
                    - src_tile_start[k];
            }
        }
        // The first update of the destination tile uses dst_mode
        copy_work(src_tile, src_tile_start, dst, dst_start,
                copy_tile_shape, scratch, dst_mode);
        // Proceed with all the rest source tiles
        std::vector<Index> src_tile_index(src_tile_index_begin);
        // Cycle through all corresponding source tiles
        for(Index j = 1; j < src_ntiles; ++j)
        {
            // Get next source tile
            ++src_tile_index[0];
            Index k = 0;
            while(src_tile_index[k] == src_tile_index_end[k])
            {
                src_tile_index[k] = src_tile_index_begin[k];
                ++k;
                ++src_tile_index[k];
            }
            // Get starting indices source and destination tiles and deduce
            // shape of copy
            std::vector<Index> dst_tile_start(ndim);
            for(Index k = 0; k < ndim; ++k)
            {
                if(src_tile_index[k] == src_tile_index_begin[k])
                {
                    src_tile_start[k] = src_start[k]
                        - src_tile_index[k]*src.basetile_shape[k];
                    dst_tile_start[k] = dst_start[k];
                }
                else
                {
                    src_tile_start[k] = 0;
                    dst_tile_start[k] = dst_start[k] - src_start[k]
                        + src_tile_index[k]*src.basetile_shape[k];
                }
                if(src_tile_index[k]+1 == src_tile_index_end[k])
                {
                    copy_tile_shape[k] = src_start[k] + copy_shape[k]
                        - src_tile_index[k]*src.basetile_shape[k]
                        - src_tile_start[k];
                }
                else
                {
                    copy_tile_shape[k] = src.basetile_shape[k]
                        - src_tile_start[k];
                }
            }
            // Call copy instruction for the tiles
            auto src_tile = src.get_tile(src_tile_index);
            copy_work(src_tile, src_tile_start, dst,
                    dst_tile_start, copy_tile_shape, scratch, STARPU_RW);
        }
    }
}

template
void copy_work(const Tensor<fp32_t> &src,
        const std::vector<Index> &src_offset, const Tile<fp32_t> &dst,
        const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch);

template
void copy_work(const Tensor<fp64_t> &src,
        const std::vector<Index> &src_offset, const Tile<fp64_t> &dst,
        const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch);

template<typename T>
void copy_work(const Tensor<T> &src,
        const std::vector<Index> &src_offset, const Tensor<T> &dst,
        const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch)
{
    Index ndim = src.ndim;
    std::vector<Index> src_start(ndim), dst_start(ndim), copy_shape(ndim);
    std::vector<Index> dst_tile_index_begin(ndim), dst_tile_index_end(ndim);
    Index dst_ntiles = 1;
    // Obtain starting indices and shape of intersection for copying
    for(Index i = 0; i < ndim; ++i)
    {
        // Do nothing if tensors do not intersect
        if((src_offset[i]+src.shape[i] <= dst_offset[i])
                or (dst_offset[i]+dst.shape[i] <= src_offset[i]))
        {
            return;
        }
        // Copy to the beginning of destination
        if(src_offset[i] < dst_offset[i])
        {
            src_start[i] = dst_offset[i] - src_offset[i]; // positive value
            dst_start[i] = 0;
            copy_shape[i] = std::min(src.shape[i]-src_start[i],
                    dst.shape[i]);
        }
        // Copy from the beginning of source
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
    // Cycle through all destination tiles
    std::vector<Index> dst_tile_index(dst_tile_index_begin);
    for(Index i = 0; i < dst_ntiles; ++i)
    {
        auto dst_tile = dst.get_tile(dst_tile_index);
        Index src_ntiles = 1;
        enum starpu_data_access_mode dst_tile_mode = STARPU_W;
        std::vector<Index> src_tile_index_begin(ndim),
            src_tile_index_end(ndim);
        // Find corresponding source tiles
        for(Index j = 0; j < ndim; ++j)
        {
            if(dst_tile_index[j] == dst_tile_index_begin[j])
            {
                src_tile_index_begin[j] = src_start[j] / src.basetile_shape[j];
                // Check if destination tile is only partially overwritten
                if(dst_tile_index[j]*dst.basetile_shape[j] != dst_start[j])
                {
                    dst_tile_mode = STARPU_RW;
                }
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
                // Check if destination tile is only partially overwritten
                if(dst_tile_index[j]*dst.basetile_shape[j]+dst_tile.shape[j]
                            != dst_start[j]+copy_shape[j])
                {
                    dst_tile_mode = STARPU_RW;
                }
            }
            else
            {
                src_tile_index_end[j] =
                    ((dst_tile_index[j]+1)*dst.basetile_shape[j]-1
                     -dst_start[j]+src_start[j]) / src.basetile_shape[j] + 1;
            }
            src_ntiles *= src_tile_index_end[j] - src_tile_index_begin[j];
        }
        // Process the first source tile separately
        auto src_tile = src.get_tile(src_tile_index_begin);
        std::vector<Index> src_tile_start(ndim), dst_tile_start(ndim),
            copy_tile_shape(ndim);
        for(Index k = 0; k < ndim; ++k)
        {
            if(dst_tile_index[k] == dst_tile_index_begin[k])
            {
                src_tile_start[k] = src_start[k]
                    - src_tile_index_begin[k]*src.basetile_shape[k];
                dst_tile_start[k] = dst_start[k]
                    - dst_tile_index[k]*dst.basetile_shape[k];
            }
            else
            {
                src_tile_start[k] = src_start[k] - dst_start[k]
                    + dst_tile_index[k]*dst.basetile_shape[k]
                    - src_tile_index_begin[k]*src.basetile_shape[k];
                dst_tile_start[k] = 0;
            }
            if(src_tile_index_begin[k]+1 == src_tile_index_end[k])
            {
                if(dst_tile_index[k]+1 == dst_tile_index_end[k])
                {
                    copy_tile_shape[k] = src_start[k] + copy_shape[k]
                        - src_tile_index_begin[k]*src.basetile_shape[k]
                        - src_tile_start[k];
                }
                else
                {
                    copy_tile_shape[k] = dst.basetile_shape[k]
                        - dst_tile_start[k];
                }
            }
            else
            {
                copy_tile_shape[k] = src.basetile_shape[k]
                    - src_tile_start[k];
            }
        }
        // If there is only one corresponding source tile
        if(src_ntiles == 1)
        {
            // Check if we can simply copy entire buffer
            if(dst_tile_mode == STARPU_W and copy_tile_shape == src_tile.shape)
            {
                starpu_data_cpy(dst_tile, src_tile, 1, nullptr, nullptr);
            }
            // Smart copying otherwise
            else
            {
                copy_work(src_tile, src_tile_start, dst_tile,
                        dst_tile_start, copy_tile_shape, scratch,
                        dst_tile_mode);
            }
        }
        else
        {
            // The first update of the destination tile uses dst_tile_mode
            copy_work(src_tile, src_tile_start, dst_tile,
                    dst_tile_start, copy_tile_shape, scratch,
                    dst_tile_mode);
            // Proceed with all the rest source tiles
            std::vector<Index> src_tile_index(src_tile_index_begin);
            // Cycle through all corresponding source tiles
            for(Index j = 1; j < src_ntiles; ++j)
            {
                // Get next source tile
                ++src_tile_index[0];
                Index k = 0;
                while(src_tile_index[k] == src_tile_index_end[k])
                {
                    src_tile_index[k] = src_tile_index_begin[k];
                    ++k;
                    ++src_tile_index[k];
                }
                // Get starting indices of source and destination tiles and
                // deduce shape of copy
                for(Index k = 0; k < ndim; ++k)
                {
                    if(src_tile_index[k] == src_tile_index_begin[k])
                    {
                        if(dst_tile_index[k] == dst_tile_index_begin[k])
                        {
                            src_tile_start[k] = src_start[k]
                                - src_tile_index[k]*src.basetile_shape[k];
                            dst_tile_start[k] = dst_start[k]
                                - dst_tile_index[k]*dst.basetile_shape[k];
                        }
                        else
                        {
                            src_tile_start[k] = src_start[k] - dst_start[k]
                                + dst_tile_index[k]*dst.basetile_shape[k]
                                - src_tile_index[k]*src.basetile_shape[k];
                            dst_tile_start[k] = 0;
                        }
                    }
                    else
                    {
                        src_tile_start[k] = 0;
                        dst_tile_start[k] = dst_start[k] - src_start[k]
                            + src_tile_index[k]*src.basetile_shape[k]
                            - dst_tile_index[k]*dst.basetile_shape[k];
                    }
                    if(src_tile_index[k]+1 == src_tile_index_end[k])
                    {
                        if(dst_tile_index[k]+1 == dst_tile_index_end[k])
                        {
                            copy_tile_shape[k] = src_start[k] + copy_shape[k]
                                - src_tile_index[k]*src.basetile_shape[k]
                                - src_tile_start[k];
                        }
                        else
                        {
                            copy_tile_shape[k] = dst.basetile_shape[k]
                                - dst_tile_start[k];
                        }
                    }
                    else
                    {
                        copy_tile_shape[k] = src.basetile_shape[k]
                            - src_tile_start[k];
                    }
                }
                // Call copy instruction for the tiles
                auto src_tile = src.get_tile(src_tile_index);
                copy_work(src_tile, src_tile_start, dst_tile,
                        dst_tile_start, copy_tile_shape, scratch, STARPU_RW);
            }
        }
        // Get out if it was the last tile
        if(i == dst_ntiles-1)
        {
            break;
        }
        // Get next tile
        ++dst_tile_index[0];
        Index k = 0;
        while(dst_tile_index[k] == dst_tile_index_end[k])
        {
            dst_tile_index[k] = dst_tile_index_begin[k];
            ++k;
            ++dst_tile_index[k];
        }
    }
}

template
void copy_work(const Tensor<fp32_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp32_t> &dst,
        const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch);

template
void copy_work(const Tensor<fp64_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp64_t> &dst,
        const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch);

} // namespace nntile

