/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/gather.cc
 * Gather operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/gather.hh"
#include <algorithm>
#include "nntile/tile/copy_intersection.hh"
#include "nntile/tile/clear.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise gather operation
/*! Gather a distributed grid of tiles into a single-tiled tensor, stored on a
 * single node.
 *
 * @param[in] src: Source tensor
 * @param[inout] dst: Destination tensor
 * */
template<typename T>
void gather_async(const Tensor<T> &src, const Tensor<T> &dst)
{
    // Check if destination is a single-tile tensor
    if(dst.grid.nelems != 1)
    {
        throw std::runtime_error("Destination must be a single-tiled tensor");
    }
    // Check if shapes match
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }
    tile::Tile<int64_t> scratch_tile({std::max<Index>(1, 2*src.ndim)});
    auto dst_tile = dst.get_tile(0);
    auto dst_tile_handle = dst.get_tile_handle(0);
    // Prohibit gather from tensor into itself (same shape implies same layout)
    if(src.get_tile_handle(0).get() == dst_tile_handle.get())
    {
        throw std::runtime_error("Cannot gather from tensor into itself");
    }
    // Treat special case of a single source tile
    if(src.grid.nelems == 1)
    {
        auto src_tile = src.get_tile(0);
        std::vector<Index> zero_offset(src.ndim, 0);
        tile::copy_intersection_async<T>(src_tile, zero_offset, dst_tile,
                zero_offset, scratch_tile);
        dst_tile_handle.mpi_flush();
        return;
    }
    // Do the slow complex copy
    Index ndim = src.ndim;
    std::vector<Index> dst_offset(ndim, 0);
    std::vector<Index> src_tile_index(ndim, 0);
    tile::clear_async<T>(dst_tile);
    // Init with the first source tile
    {
        auto src_tile = src.get_tile(0);
        std::vector<Index> src_offset(ndim, 0);
        tile::copy_intersection_async<T>(src_tile, src_offset, dst_tile,
                dst_offset, scratch_tile);
    }
    // Cycle through all other source tiles
    for(Index i = 1; i < src.grid.nelems; ++i)
    {
        // Get next tile index and corresponding offset
        ++src_tile_index[0];
        Index k = 0;
        while(src_tile_index[k] == src.grid.shape[k])
        {
            src_tile_index[k] = 0;
            ++k;
            ++src_tile_index[k];
        }
        std::vector<Index> src_offset(ndim);
        for(Index k = 0; k < ndim; ++k)
        {
            src_offset[k] = src_tile_index[k] * src.basetile_shape[k];
        }
        auto src_tile = src.get_tile(i);
        tile::copy_intersection_async<T>(src_tile, src_offset, dst_tile,
                dst_offset, scratch_tile);
    }
    // Flush cache for the output tile on every node
    dst_tile_handle.mpi_flush();
}

//! Blocking version of tensor-wise gather operation
/*! Gather a distributed grid of tiles into a single-tiled tensor, stored on a
 * single node.
 *
 * @param[in] src: Source tensor
 * @param[inout] dst: Destination tensor
 * */
template<typename T>
void gather(const Tensor<T> &src, const Tensor<T> &dst)
{
    gather_async<T>(src, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void gather_async<fp32_t>(const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &dst);

template
void gather_async<fp64_t>(const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &dst);

template
void gather_async<int64_t>(const Tensor<int64_t> &src,
        const Tensor<int64_t> &dst);

template
void gather_async<bool_t>(const Tensor<bool_t> &src,
        const Tensor<bool_t> &dst);

template
void gather_async<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void gather_async<fp32_fast_fp16_t>(const Tensor<fp32_fast_fp16_t> &src,
                              const Tensor<fp32_fast_fp16_t> &dst);

template
void gather_async<fp32_fast_bf16_t>(const Tensor<fp32_fast_bf16_t> &src,
                              const Tensor<fp32_fast_bf16_t> &dst);

template
void gather_async<bf16_t>(const Tensor<bf16_t> &src,
        const Tensor<bf16_t> &dst);

template
void gather_async<fp16_t>(const Tensor<fp16_t> &src,
        const Tensor<fp16_t> &dst);

// Explicit instantiation
template
void gather<fp32_t>(const Tensor<fp32_t> &src, const Tensor<fp32_t> &dst);

template
void gather<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &src,
                              const Tensor<fp32_fast_tf32_t> &dst);

template
void gather<fp32_fast_fp16_t>(const Tensor<fp32_fast_fp16_t> &src,
                              const Tensor<fp32_fast_fp16_t> &dst);

template
void gather<fp32_fast_bf16_t>(const Tensor<fp32_fast_bf16_t> &src,
                              const Tensor<fp32_fast_bf16_t> &dst);

template
void gather<fp64_t>(const Tensor<fp64_t> &src, const Tensor<fp64_t> &dst);

template
void gather<int64_t>(const Tensor<int64_t> &src, const Tensor<int64_t> &dst);

template
void gather<bool_t>(const Tensor<bool_t> &src, const Tensor<bool_t> &dst);

template
void gather<bf16_t>(const Tensor<bf16_t> &src, const Tensor<bf16_t> &dst);

template
void gather<fp16_t>(const Tensor<fp16_t> &src, const Tensor<fp16_t> &dst);

} // namespace nntile::tensor
