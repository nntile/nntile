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

#include "nntile/tensor/norm.hh"
#include "nntile/tile/norm.hh"
#include "nntile/tile/copy.hh"

namespace nntile
{

template<typename T>
static void cpu_sum_ssq_accumulate(void *buffers[], void *cl_args)
{
    Index nelems;
    starpu_codelet_unpack_args(cl_args, &nelems);
    const T *src = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    T *dst = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
    for(Index i = 0; i < nelems; i += 3)
    {
        // If maximal absolute value is 0 do no update to avoid division by 0
        if(src[i+1] == 0)
        {
            continue;
        }
        // Now src[i+1]>0
        dst[i] += src[i];
        if(dst[i+1] > src[i+1])
        {
            T tmp = src[i+1] / dst[i+1];
            dst[i+2] += src[i+2] * tmp * tmp;
        }
        else
        {
            // No division by 0 here since src[i+1]>0
            T tmp = dst[i+1] / src[i+1];
            dst[i+1] = src[i+1];
            dst[i+2] = dst[i+2]*tmp*tmp + src[i+2];
        }
    }
}

template<typename T>
void norm_sum_ssq_accumulate_async(const Tile<T> &sum_ssq,
        const Tile<T> &sum_ssq_total)
{
    static starpu_codelet codelet_sum_ssq_accumulate =
    {
        .cpu_funcs = {cpu_sum_ssq_accumulate<T>},
        .nbuffers = 2,
        .modes = {STARPU_R, STARPU_RW}
    };
    // Check inputs
    if(sum_ssq.ndim != sum_ssq_total.ndim)
    {
        throw std::runtime_error("sum_ssq.ndim != sum_ssq_total.ndim");
    }
    Index ndim = sum_ssq.ndim;
    for(Index i = 0; i < ndim; ++i)
    {
        if(sum_ssq.shape[i] != sum_ssq_total.shape[i])
        {
            throw std::runtime_error("sum_ssq.shape[i] != "
                    "sum_ssq_total.shape[i]");
        }
    }
    // Insert task
    starpu_task_insert(&codelet_sum_ssq_accumulate,
            STARPU_VALUE, &(sum_ssq.nelems), sizeof(sum_ssq.nelems),
            STARPU_R, static_cast<starpu_data_handle_t>(sum_ssq),
            STARPU_RW, static_cast<starpu_data_handle_t>(sum_ssq_total),
            0);
}

template
void norm_sum_ssq_accumulate_async(const Tile<fp32_t> &sum_ssq,
        const Tile<fp32_t> &sum_ssq_total);

template
void norm_sum_ssq_accumulate_async(const Tile<fp64_t> &sum_ssq,
        const Tile<fp64_t> &sum_ssq_total);

template<typename T>
void norm_sum_ssq_async(const Tensor<T> &src, const Tensor<T> &sum_ssq,
        const Tensor<T> &sum_ssq_work, const std::vector<Index> &axes)
{
    // Check dimensions
    if(src.ndim+1 != sum_ssq.ndim+axes.size())
    {
        throw std::runtime_error("src.ndim+1 != sum_ssq.ndim+axes.size()");
    }
    if(src.ndim+1 != sum_ssq_work.ndim)
    {
        throw std::runtime_error("src.ndim+1 != sum_ssq.ndim+axes.size()");
    }
    // Treat special case of src.ndim=0
    if(src.ndim == 0)
    {
        throw std::runtime_error("Scalar input makes no sense");
    }
    // Treat special case of empty axes
    if(axes.size() == 0)
    {
        throw std::runtime_error("Empty axes");
    }
    // Check axes
    if(axes[0] < 0)
    {
        throw std::runtime_error("axes[0] < 0");
    }
    if(axes[axes.size()-1] >= src.ndim)
    {
        throw std::runtime_error("axes[axes.size()-1] >= src.ndim");
    }
    for(Index i = 1; i < axes.size(); ++i)
    {
        if(axes[i] <= axes[i-1])
        {
            throw std::runtime_error("axes[i] <= axes[i-1]");
        }
    }
    // Check shapes of src and sum_ssq
    if(sum_ssq.shape[0] != 3)
    {
        throw std::runtime_error("sum_ssq.shape[0] != 3");
    }
    if(sum_ssq.basetile_shape[0] != 3)
    {
        throw std::runtime_error("sum_ssq.basetile_shape[0] != 3");
    }
    if(sum_ssq_work.shape[0] != 3)
    {
        throw std::runtime_error("sum_ssq_work.shape[0] != 3");
    }
    if(sum_ssq_work.basetile_shape[0] != 3)
    {
        throw std::runtime_error("sum_ssq_work.basetile_shape[0] != 3");
    }
    // Number of checked items in axes
    Index nchecked_axes = 0;
    for(Index i = 0; i < src.ndim; ++i)
    {
        if(nchecked_axes < axes.size() and i == axes[nchecked_axes])
        {
            if(sum_ssq_work.shape[i+1] != src.grid.shape[i])
            {
                throw std::runtime_error("sum_ssq_work.shape[i+1] != "
                        "src.grid.shape[i]");
            }
            if(sum_ssq_work.basetile_shape[i+1] != 1)
            {
                throw std::runtime_error("sum_ssq_work.basetile_shape[i+1] "
                        "!= 1");
            }
            ++nchecked_axes;
            continue;
        }
        if(src.shape[i] != sum_ssq.shape[i-nchecked_axes+1])
        {
            throw std::runtime_error("src.shape[i] != "
                    "sum_ssq.shape[i-nchecked_axes+1]");
        }
        if(src.basetile_shape[i] != sum_ssq.basetile_shape[i-nchecked_axes+1])
        {
            throw std::runtime_error("src.basetile_shape[i] != "
                    "sum_ssq.basetile_shape[i-nchecked_axes+1]");
        }
        if(src.shape[i] != sum_ssq_work.shape[i+1])
        {
            throw std::runtime_error("src.shape[i] != "
                    "sum_ssq_work.shape[i+1]");
        }
        if(src.basetile_shape[i] != sum_ssq_work.basetile_shape[i+1])
        {
            throw std::runtime_error("src.basetile_shape[i] != "
                    "sum_ssq_work.basetile_shape[i+1]");
        }
    }
    // Non-slice axes
    std::vector<Index> sum_ssq_axes;
    sum_ssq_axes.reserve(sum_ssq.ndim-1);
    Index j = 0;
    for(Index i = 0; i < src.ndim; ++i)
    {
        if(j == axes.size() or axes[j] != i)
        {
            sum_ssq_axes.push_back(i);
        }
        else
        {
            ++j;
        }
    }
    // Compute sum and sum of squares for each tile in grid of src tensor
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        // Linear offsets of corresponding tiles of work and src tensors are
        // equal
        auto work_tile = sum_ssq_work.get_tile(i);
        // Reshape
        TileTraits &work_tile_traits = work_tile;
        std::vector<Index> new_shape = {3};
        new_shape.reserve(sum_ssq.ndim-1);
        Index nchecked_axes = 0;
        for(Index j = 0; j < src.ndim; ++j)
        {
            if(nchecked_axes < axes.size() and j == axes[nchecked_axes])
            {
                ++nchecked_axes;
            }
            else
            {
                new_shape.push_back(work_tile.shape[j+1]);
            }
        }
        work_tile_traits = TileTraits(new_shape);
        // Launch per-tile kernel
        norm_sum_ssq_async(src.get_tile(i), work_tile, axes);
    }
    // Get number of slices and size of each slice
    Index slice_size = 1;
    for(Index i = 0; i < axes.size(); ++i)
    {
        slice_size *= src.grid.shape[axes[i]];
    }
    Index nslices = src.grid.nelems / slice_size;
    // Accumulate results for the first slice
    auto dst_tile = sum_ssq_work.get_tile(0);
    std::vector<Index> src_tile_index(sum_ssq_work.ndim);
    for(Index j = 1; j < slice_size; ++j)
    {
        ++src_tile_index[axes[0]+1];
        Index k = 0;
        while(src_tile_index[axes[k]+1] == src.grid.shape[axes[k]])
        {
            src_tile_index[axes[k]+1] = 0;
            ++k;
            ++src_tile_index[axes[k]+1];
        }
        Index src_tile_offset = sum_ssq_work.grid.index_to_linear(
                src_tile_index);
        auto src_tile = sum_ssq_work.get_tile(src_tile_offset);
        norm_sum_ssq_accumulate_async(src_tile, dst_tile);
    }
    // Create a reshaped tile of sum_ssq_work
    TileTraits &dst_tile_traits = dst_tile;
    std::vector<Index> new_shape = {3};
    new_shape.reserve(sum_ssq.ndim-1);
    nchecked_axes = 0;
    for(Index j = 0; j < src.ndim; ++j)
    {
        if(nchecked_axes < axes.size() and j == axes[nchecked_axes])
        {
            ++nchecked_axes;
        }
        else
        {
            new_shape.push_back(dst_tile.shape[j+1]);
        }
    }
    dst_tile_traits = TileTraits(new_shape);
    copy_intersection(dst_tile, sum_ssq.get_tile(0));
    // Other slices
    std::vector<Index> dst_tile_index(sum_ssq_work.ndim);
    std::vector<Index> sum_ssq_tile_index(sum_ssq.ndim);
    for(Index i = 1; i < nslices; ++i)
    {
        ++dst_tile_index[sum_ssq_axes[0]+1];
        ++sum_ssq_tile_index[1];
        Index j = 0;
        while(sum_ssq_tile_index[j+1] == sum_ssq.grid.shape[j+1])
        {
            dst_tile_index[sum_ssq_axes[j]+1] = 0;
            sum_ssq_tile_index[j+1] = 0;
            ++j;
            ++dst_tile_index[sum_ssq_axes[j]+1];
            ++sum_ssq_tile_index[j+1];
        }
        Index dst_tile_offset = sum_ssq_work.grid.index_to_linear(
                dst_tile_index);
        auto dst_tile = sum_ssq_work.get_tile(dst_tile_offset);
        std::vector<Index> src_tile_index(dst_tile_index);
        for(Index j = 1; j < slice_size; ++j)
        {
            ++src_tile_index[axes[0]+1];
            Index k = 0;
            while(src_tile_index[axes[k]+1] == src.grid.shape[axes[k]])
            {
                src_tile_index[axes[k]+1] = 0;
                ++k;
                ++src_tile_index[axes[k]+1];
            }
            Index src_tile_offset = sum_ssq_work.grid.index_to_linear(
                    src_tile_index);
            auto src_tile = sum_ssq_work.get_tile(src_tile_offset);
            norm_sum_ssq_accumulate_async(src_tile, dst_tile);
        }
        // Create a reshaped tile of sum_ssq_work
        TileTraits &dst_tile_traits = dst_tile;
        std::vector<Index> new_shape = {3};
        new_shape.reserve(sum_ssq.ndim-1);
        Index nchecked_axes = 0;
        for(Index j = 0; j < src.ndim; ++j)
        {
            if(nchecked_axes < axes.size() and j == axes[nchecked_axes])
            {
                ++nchecked_axes;
            }
            else
            {
                new_shape.push_back(dst_tile.shape[j+1]);
            }
        }
        dst_tile_traits = TileTraits(new_shape);
        copy_intersection(dst_tile, sum_ssq.get_tile(sum_ssq_tile_index));
    }
}

template
void norm_sum_ssq_async(const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &sum_ssq, const Tensor<fp32_t> &sum_ssq_work,
        const std::vector<Index> &axes);

template
void norm_sum_ssq_async(const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &sum_ssq, const Tensor<fp64_t> &sum_ssq_work,
        const std::vector<Index> &axes);

} // namespace nntile

