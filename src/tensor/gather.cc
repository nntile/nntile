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
#include "nntile/starpu/subcopy.hh"
#include "nntile/starpu/copy.hh"

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
    // Treat special case of a source destination tile
    int mpi_rank = starpu_mpi_world_rank();
    auto dst_tile_handle = dst.get_tile_handle(0);
    auto dst_tile_traits = dst.get_tile_traits(0);
    int dst_tile_rank = dst_tile_handle.mpi_get_rank();
    int ret;
    if(src.grid.nelems == 1)
    {
        auto src_tile_handle = src.get_tile_handle(0);
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
    std::vector<Index> src_tile_index(ndim);
    // Init with the first source tile
    auto src_first_tile_handle = src.get_tile_handle(0);
    // Transfer first source tile to dest node
    src_first_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
    // Execute on dest tile
    if(mpi_rank == dst_tile_rank)
    {
        auto src_first_tile_traits = src.get_tile_traits(0);
        starpu::subcopy::submit<T>(ndim, src_tile_start,
                src_first_tile_traits.stride, dst_tile_start,
                dst_tile_traits.stride, src_first_tile_traits.shape,
                src_first_tile_handle, dst_tile_handle, scratch, STARPU_W);
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
        auto src_tile_handle = src.get_tile_handle(i);
        // Transfer source tile to dest node
        src_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        // Execute on dest tile
        if(mpi_rank == dst_tile_rank)
        {
            auto src_tile_traits = src.get_tile_traits(i);
            for(Index k = 0; k < ndim; ++k)
            {
                dst_tile_start[k] = src_tile_index[k] * src.basetile_shape[k];
            }
            starpu::subcopy::submit<T>(ndim, src_tile_start,
                    src_tile_traits.stride, dst_tile_start,
                    dst_tile_traits.stride, src_tile_traits.shape,
                    src_tile_handle, dst_tile_handle, scratch, STARPU_RW);
        }
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
void gather_async<bf16_t>(const Tensor<bf16_t> &src,
        const Tensor<bf16_t> &dst);

// Explicit instantiation
template
void gather<fp32_t>(const Tensor<fp32_t> &src, const Tensor<fp32_t> &dst);

template
void gather<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &src,
                              const Tensor<fp32_fast_tf32_t> &dst);

template
void gather<fp64_t>(const Tensor<fp64_t> &src, const Tensor<fp64_t> &dst);

template
void gather<int64_t>(const Tensor<int64_t> &src, const Tensor<int64_t> &dst);

template
void gather<bool_t>(const Tensor<bool_t> &src, const Tensor<bool_t> &dst);

template
void gather<bf16_t>(const Tensor<bf16_t> &src, const Tensor<bf16_t> &dst);

} // namespace nntile::tensor
