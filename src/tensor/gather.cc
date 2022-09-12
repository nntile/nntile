/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/gather.cc
 * Gather operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-12
 * */

#include "nntile/tensor/gather.hh"
#include "nntile/starpu/subcopy.hh"

namespace nntile
{
namespace tensor
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
    // Treat special case of a source destination tile
    int mpi_rank = starpu_mpi_world_rank();
    auto dst_tile_handle = dst.get_tile_handle(0);
    auto dst_tile_traits = dst.get_tile_traits(0);
    int dst_tile_rank = starpu_mpi_data_get_rank(dst_tile_handle);
    int ret;
    if(src.grid.nelems == 1)
    {
        auto src_tile_handle = src.get_tile_handle(0);
        int src_tile_rank = starpu_mpi_data_get_rank(src_tile_handle);
        int tile_tag = starpu_mpi_data_get_tag(src_tile_handle);
        // Init send for owner of source tile
        if(mpi_rank == src_tile_rank)
        {
            // If both source and destination are owned by the same node
            if(mpi_rank == dst_tile_rank)
            {
                ret = starpu_data_cpy(dst_tile_handle, src_tile_handle, 1,
                        nullptr, nullptr);
                if(ret != 0)
                {
                    throw std::runtime_error("Error in starpu_data_cpy");
                }
            }
            else
            {
                ret = starpu_mpi_isend_detached(src_tile_handle, dst_tile_rank,
                        tile_tag, MPI_COMM_WORLD, nullptr, nullptr);
                if(ret != 0)
                {
                    throw std::runtime_error("Error in starpu_mpi_isend_"
                            "detached");
                }
            }
        }
        // Init receive for owner of destination tile
        else if(mpi_rank == dst_tile_rank)
        {
            ret = starpu_mpi_irecv_detached(dst_tile_handle, src_tile_rank,
                    tile_tag, MPI_COMM_WORLD, nullptr, nullptr);
            if(ret != 0)
            {
                throw std::runtime_error("Error in starpu_mpi_irecv_detached");
            }
        }
        return;
    }
    // Do the slow complex copy
    // Temporary buffer for indexing, that is allocated per-worker when needed
    Index ndim = src.ndim;
    StarpuVariableHandle scratch(2 * ndim * sizeof(Index));
    // We define starting coordinates and shapes for all complex copies of
    // tiles
    std::vector<Index> src_tile_start(ndim), dst_tile_start(ndim);
    // Cycle through all source tiles
    std::vector<Index> src_tile_index(ndim);
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        auto src_tile_traits = src.get_tile_traits(i);
        auto src_tile_handle = src.get_tile_handle(i);
        int src_tile_rank = starpu_mpi_data_get_rank(src_tile_handle);
        auto tile_tag = starpu_mpi_data_get_tag(src_tile_handle);
        for(Index k = 0; k < ndim; ++k)
        {
            dst_tile_start[k] = src_tile_index[k] * src.basetile_shape[k];
        }
        // Init send of destination tile for owner of source tile
        if(mpi_rank == src_tile_rank)
        {
            // Perform MPI copy only if destination node is different
            if(mpi_rank != dst_tile_rank)
            {
                ret = starpu_mpi_isend_detached(src_tile_handle, dst_tile_rank,
                        tile_tag, MPI_COMM_WORLD, nullptr, nullptr);
                if(ret != 0)
                {
                    throw std::runtime_error("Error in starpu_mpi_isend_"
                            "detached");
                }
            }
        }
        // Init receive of source tile for owner of destination tile
        if(mpi_rank == dst_tile_rank)
        {
            // Perform MPI copy only if source node is different
            if(mpi_rank != src_tile_rank)
            {
                ret = starpu_mpi_irecv_detached(src_tile_handle, src_tile_rank,
                        tile_tag, MPI_COMM_WORLD, nullptr, nullptr);
                if(ret != 0)
                {
                    throw std::runtime_error("Error in starpu_mpi_irecv_"
                            "detached");
                }
            }
            // Perform complex copy
            starpu::subcopy::submit<T>(ndim, src_tile_start,
                    src_tile_traits.stride, dst_tile_start,
                    dst_tile_traits.stride, src_tile_traits.shape,
                    src_tile_handle, dst_tile_handle, scratch, STARPU_W);
        }
        // Get out if it was the last tile
        if(i == src.grid.nelems-1)
        {
            break;
        }
        // Get next tile
        ++src_tile_index[0];
        Index k = 0;
        while(src_tile_index[k] == src.grid.shape[k])
        {
            src_tile_index[k] = 0;
            ++k;
            ++src_tile_index[k];
        }
    }
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
void gather<fp32_t>(const Tensor<fp32_t> &src, const Tensor<fp32_t> &dst);

template
void gather<fp64_t>(const Tensor<fp64_t> &src, const Tensor<fp64_t> &dst);

} // namespace tensor
} // namespace nntile

