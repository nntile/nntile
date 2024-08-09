/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/scatter.cc
 * Scatter operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/scatter.hh"
#include "nntile/starpu/subcopy.hh"
#include "nntile/starpu/copy.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise scatter operation
/*! Scatter a single-tiled tensor, stored on a single node, into a distributed
 * grid of tiles.
 *
 * @param[in] src: Source tensor
 * @param[inout] dst: Destination tensor
 * */
template<typename T>
void scatter_async(const Tensor<T> &src, const Tensor<T> &dst)
{
    // Check if source is a single-tile tensor
    if(src.grid.nelems != 1)
    {
        throw std::runtime_error("Source must be a single-tiled tensor");
    }
    // Check if shapes match
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }
    // Treat special case of a single destination tile
    int mpi_rank = starpu_mpi_world_rank();
    auto src_tile_handle = src.get_tile_handle(0);
    auto src_tile_traits = src.get_tile_traits(0);
    int src_tile_rank = src_tile_handle.mpi_get_rank();
    int ret;
    if(dst.grid.nelems == 1)
    {
        auto dst_tile_handle = dst.get_tile_handle(0);
        int dst_tile_rank = dst_tile_handle.mpi_get_rank();
        // Transfer source tile to dest node
        src_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        // Execute on destination node
        if(mpi_rank == dst_tile_rank)
        {
            starpu::copy::submit(src_tile_handle, dst_tile_handle);
        }
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
        return;
    }
    // Do the slow complex copy
    // Temporary buffer for indexing, that is allocated per-worker when needed
    Index ndim = src.ndim;
    starpu::VariableHandle scratch(2*ndim*sizeof(int64_t), STARPU_SCRATCH);
    // We define starting coordinates and shapes for all complex copies of
    // tiles
    std::vector<Index> src_tile_start(ndim), dst_tile_start(ndim);
    // Cycle through all destination tiles
    std::vector<Index> dst_tile_index(ndim);
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        auto dst_tile_handle = dst.get_tile_handle(i);
        int dst_tile_rank = dst_tile_handle.mpi_get_rank();
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
        // Execute on source node and then send result
        if(mpi_rank == src_tile_rank)
        {
            auto dst_tile_traits = dst.get_tile_traits(i);
            for(Index k = 0; k < ndim; ++k)
            {
                src_tile_start[k] = dst_tile_index[k] * dst.basetile_shape[k];
            }
            // Perform complex copy into local dst_tile_handle
            starpu::subcopy::submit<T>(ndim, src_tile_start,
                    src_tile_traits.stride, dst_tile_start,
                    dst_tile_traits.stride, dst_tile_traits.shape,
                    src_tile_handle, dst_tile_handle, scratch, STARPU_W);
            // Perform MPI copy only if destination node is different
            auto tile_tag = dst_tile_handle.mpi_get_tag();
            if(mpi_rank != dst_tile_rank)
            {
                // No need to check for cached send, as output was just updated
                //ret = starpu_mpi_isend_detached(
                //        static_cast<starpu_data_handle_t>(dst_tile_handle),
                //        dst_tile_rank, tile_tag, MPI_COMM_WORLD, nullptr,
                //        nullptr);
                //if(ret != 0)
                //{
                //    throw std::runtime_error("Error in starpu_mpi_isend_"
                //            "detached");
                //}
            }
        }
        // Init receive of source tile for owner of destination tile
        else if(mpi_rank == dst_tile_rank)
        {
            auto tile_tag = dst_tile_handle.mpi_get_tag();
            // No need to check for cached recv, as output was just updated
            //ret = starpu_mpi_irecv_detached(
            //        static_cast<starpu_data_handle_t>(dst_tile_handle),
            //        src_tile_rank, tile_tag, MPI_COMM_WORLD, nullptr,
            //        nullptr);
            //if(ret != 0)
            //{
            //    throw std::runtime_error("Error in starpu_mpi_irecv_"
            //            "detached");
            //}
        }
        // Get out if it was the last tile
        if(i == dst.grid.nelems-1)
        {
            break;
        }
        // Get next tile
        ++dst_tile_index[0];
        Index k = 0;
        while(dst_tile_index[k] == dst.grid.shape[k])
        {
            dst_tile_index[k] = 0;
            ++k;
            ++dst_tile_index[k];
        }
    }
}

//! Blocking version of tensor-wise scatter operation
/*! Scatter a single-tiled tensor, stored on a single node, into a distributed
 * grid of tiles.
 *
 * @param[in] src: Source tensor
 * @param[inout] dst: Destination tensor
 * */
template<typename T>
void scatter(const Tensor<T> &src, const Tensor<T> &dst)
{
    scatter_async<T>(src, dst);
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
    starpu_task_wait_for_all();
}

// Explicit instantiation
//template
//void scatter_async<fp16_t>(const Tensor<fp16_t> &src,
//        const Tensor<fp16_t> &dst);

template
void scatter_async<fp32_t>(const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &dst);

template
void scatter_async<fp64_t>(const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &dst);

template
void scatter_async<int64_t>(const Tensor<int64_t> &src,
        const Tensor<int64_t> &dst);

template
void scatter_async<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &src,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void scatter_async<bool_t>(const Tensor<bool_t> &src,
        const Tensor<bool_t> &dst);

template
void scatter_async<bf16_t>(const Tensor<bf16_t> &src,
        const Tensor<bf16_t> &dst);

// Explicit instantiation
//template
//void scatter<fp16_t>(const Tensor<fp16_t> &src, const Tensor<fp16_t> &dst);

template
void scatter<fp32_t>(const Tensor<fp32_t> &src, const Tensor<fp32_t> &dst);

template
void scatter<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &src,
                               const Tensor<fp32_fast_tf32_t> &dst);

template
void scatter<fp64_t>(const Tensor<fp64_t> &src, const Tensor<fp64_t> &dst);

template
void scatter<int64_t>(const Tensor<int64_t> &src, const Tensor<int64_t> &dst);

template
void scatter<bool_t>(const Tensor<bool_t> &src, const Tensor<bool_t> &dst);

template
void scatter<bf16_t>(const Tensor<bf16_t> &src, const Tensor<bf16_t> &dst);

} // namespace nntile::tensor
