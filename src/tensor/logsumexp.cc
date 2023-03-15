/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/logsumexp.cc
 * Log sum of exponents of Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-03-15
 * */

#include "nntile/tensor/logsumexp.hh"
#include "nntile/starpu/logsumexp.hh"
#include "nntile/starpu/clear.hh"

namespace nntile
{
namespace tensor
{

//! Compute max and sum of exponents of slices along given axis
template<typename T>
void logsumexp_async(const Tensor<T> &src, const Tensor<T> &dst)
{
    // Check dimensions
    if(src.ndim-1 != dst.ndim)
    {
        throw std::runtime_error("src.ndim != dst.ndim");
    }
    // Treat special case of src.ndim=0
    if(src.ndim == 0)
    {
        throw std::runtime_error("Scalar input makes no sense");
    }
    // Check shapes of src and dst
    if(src.shape[0] != 2)
    {
        throw std::runtime_error("src.shape[0] != 2");
    }
    for(Index i = 0; i < src.ndim-1; ++i)
    {
        if(src.shape[i+1] != dst.shape[i])
        {
            throw std::runtime_error("src.shape[i+1] != dst.shape[i]");
        }
        if(src.basetile_shape[i+1] != dst.basetile_shape[i])
        {
            throw std::runtime_error("src.basetile_shape[i+1] != "
                    "dst.basetile_shape[i]");
        }
    }
    // Do actual calculations
    int mpi_rank = starpu_mpi_world_rank();
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        // Clean up destination tile on dest node
        auto dst_tile_handle = dst.get_tile_handle(i);
        int dst_tile_rank = dst_tile_handle.mpi_get_rank();
        if(mpi_rank == dst_tile_rank)
        {
            starpu::clear::submit(dst_tile_handle);
        }
        // Obtain indices of applicable source tiles
        // auto dst_tile_index = dst.grid.linear_to_index(i);
        // Launch kernel for each appropriate source tile
        auto dst_tile_traits = dst.get_tile_traits(i);

        auto src_tile_handle = src.get_tile_handle(i);
        // int src_tile_rank = src_tile_handle.mpi_get_rank();
        // Transfer data
        src_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        // Execute on destination node
        if (mpi_rank == dst_tile_rank)
        {
            // Insert task
            starpu::logsumexp::submit<T>(dst_tile_traits.nelems, src_tile_handle,
                    dst_tile_handle);
        }


        // for(Index j = 0; j < src.grid.shape[axis]; ++j)
        // {
        //     src_tile_index[axis] = j;
        //     Index src_tile_offset = src.grid.index_to_linear(src_tile_index);
        //     auto src_tile_handle = src.get_tile_handle(src_tile_offset);
        //     // int src_tile_rank = src_tile_handle.mpi_get_rank();
        //     // Transfer data
        //     src_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        //     // Execute on destination node
        //     if(mpi_rank == dst_tile_rank)
        //     {
        //         // Insert task
        //         starpu::logsumexp::submit<T>(src_tile_handle,
        //                 dst_tile_handle);
        //     }
        // }
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}


template<typename T>
void logsumexp(const Tensor<T> &src, const Tensor<T> &dst)
{
    logsumexp_async<T>(src, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void logsumexp_async<fp32_t>(const Tensor<fp32_t> &src, const Tensor<fp32_t> &dst);

template
void logsumexp_async<fp64_t>(const Tensor<fp64_t> &src, const Tensor<fp64_t> &dst);

// Explicit instantiation
template
void logsumexp<fp32_t>(const Tensor<fp32_t> &src, const Tensor<fp32_t> &dst);

template
void logsumexp<fp64_t>(const Tensor<fp64_t> &src, const Tensor<fp64_t> &dst);

} // namespace tensor
} // namespace nntile
