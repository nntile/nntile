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
        copy_intersection_work_ndim0(src, dst.get_tile(0));
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
    std::vector<Index> src_start(ndim), dst_start(ndim), copy_shape(ndim);
    std::vector<Index> src_tile_index_begin(ndim), src_tile_index_end(ndim);
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
//    // Print some info
//    std::cout << "dst tiles(";
//    for(Index k = 0; k < ndim; ++k)
//    {
//        std::cout << dst_tile_index_begin[k] << ":" << dst_tile_index_end[k] << ",";
//    }
//    std::cout << ")\n";
//    // Print full copy info
//    std::cout << "Full dst(";
//    for(Index k = 0; k < ndim; ++k)
//    {
//        std::cout << "0:" << dst.shape[k] << ",";
//    }
//    std::cout << ")[";
//    for(Index k = 0; k < ndim; ++k)
//    {
//        std::cout << dst_start[k] << ":" << dst_start[k]+copy_shape[k] << ",";
//    }
//    std::cout << "]\n";
//    std::cout << "Full src(";
//    for(Index k = 0; k < ndim; ++k)
//    {
//        std::cout << "0:" << src.shape[k] << ",";
//    }
//    std::cout << ")[";
//    for(Index k = 0; k < ndim; ++k)
//    {
//        std::cout << src_start[k] << ":" << src_start[k]+copy_shape[k] << ",";
//    }
//    std::cout << "]\n";
    // Cycle through all destination tiles
    std::vector<Index> dst_tile_index(dst_tile_index_begin);
    for(Index i = 0; i < dst_ntiles; ++i)
    {
        Index src_ntiles = 1;
        // Find corresponding source tiles
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
//        // Print some info
//        std::cout << "src tiles(";
//        for(Index k = 0; k < ndim; ++k)
//        {
//            std::cout << src_tile_index_begin[k] << ":" << src_tile_index_end[k] << ",";
//        }
//        std::cout << ")\n";
//        std::cout << "dst(";
//        for(Index k = 0; k < ndim; ++k)
//        {
//            std::cout << dst_tile_index[k]*dst.basetile_shape[k] << ":";
//            if(dst_tile_index[k]+1 == dst.grid.shape[k])
//            {
//                std::cout << dst.shape[k];
//            }
//            else
//            {
//                std::cout << (dst_tile_index[k]+1)*dst.basetile_shape[k];
//            }
//            std::cout << ",";
//        }
//        std::cout << ")\n";
        auto dst_tile = dst.get_tile(dst_tile_index);
        std::vector<Index> src_tile_index(src_tile_index_begin);
        // Cycle through all corresponding source tiles
        for(Index j = 0; j < src_ntiles; ++j)
        {
            std::vector<Index> src_tile_start(ndim), dst_tile_start(ndim),
                copy_tile_shape(ndim);
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
                        src_tile_start[k] = src_start[k]
                            + dst_tile_index[k]*dst.basetile_shape[k]
                            - dst_start[k]
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
                        copy_tile_shape[k] = 0
                            + dst.basetile_shape[k]
                            - dst_tile_start[k];
                    }
                }
                else
                {
                    copy_tile_shape[k] = 0
                        + src.basetile_shape[k]
                        - src_tile_start[k];
                }
            }
//            // Print some info
//            std::cout << "from src(";
//            for(Index k = 0; k < ndim; ++k)
//            {
//                std::cout << src_tile_index[k]*src.basetile_shape[k] << ":";
//                if(src_tile_index[k]+1 == src.grid.shape[k])
//                {
//                    std::cout << src.shape[k];
//                }
//                else
//                {
//                    std::cout << (src_tile_index[k]+1)*src.basetile_shape[k];
//                }
//                std::cout << ",";
//            }
//            std::cout << ")[";
//            for(Index k = 0; k < ndim; ++k)
//            {
//                std::cout << src_tile_start[k] << ":"
//                    << src_tile_start[k]+copy_tile_shape[k] << ",";
//            }
//            std::cout << "] to dst(";
//            for(Index k = 0; k < ndim; ++k)
//            {
//                std::cout << dst_tile_index[k]*dst.basetile_shape[k] << ":";
//                if(dst_tile_index[k]+1 == dst.grid.shape[k])
//                {
//                    std::cout << dst.shape[k];
//                }
//                else
//                {
//                    std::cout << (dst_tile_index[k]+1)*dst.basetile_shape[k];
//                }
//                std::cout << ",";
//            }
//            std::cout << ")[";
//            for(Index k = 0; k < ndim; ++k)
//            {
//                std::cout << dst_tile_start[k] << ":"
//                    << dst_tile_start[k]+copy_tile_shape[k] << ",";
//            }
//            std::cout << "]\n";
            // Call copy for the tiles
            auto src_tile = src.get_tile(src_tile_index);
            copy_intersection_work(src_tile, src_tile_start, dst_tile,
                    dst_tile_start, copy_tile_shape, scratch, STARPU_W);
            // Get out if it was the last source tile
            if(j == src_ntiles-1)
            {
                break;
            }
            // Get next tile
            ++src_tile_index[0];
            Index k = 0;
            while(src_tile_index[k] == src_tile_index_end[k])
            {
                src_tile_index[k] = src_tile_index_begin[k];
                ++k;
                ++src_tile_index[k];
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

