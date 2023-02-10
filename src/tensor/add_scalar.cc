/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/add_scalar.cc
 * Add scalar to elements from Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-01-10
 * */

#include "nntile/tensor/add_scalar.hh"
#include "nntile/starpu/add_scalar.hh"

namespace nntile
{
namespace tensor
{
//! Asynchronous tensor-wise add_scalar operation
/*! @param[in] alpha: Input scalar value 
 * @param[inout] src: Input and output tensor for the add_scalar operation
 * */
template<typename T>
void add_scalar_async(T alpha, const Tensor<T> &src)
{
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    // Launch all the required tasks
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        // Get handle for corresponding tiles of src and dst
        auto src_tile_handle = src.get_tile_handle(i);
        // MPI rank of the destination tile
        int src_tile_rank = src_tile_handle.mpi_get_rank();
        // Transfer data
        src_tile_handle.mpi_transfer(src_tile_rank, mpi_rank);
        // Execute only on destination node
        if(mpi_rank == src_tile_rank)
        {
            auto traits = src.get_tile_traits(i);
            starpu::add_scalar::submit<T>(alpha, traits.nelems,
                    src_tile_handle);
        }
        // Flush cache for the output tile on every node
        src_tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise add_scalar operation
/*! @param[in] alpha: Input scalar value 
 * @param[inout] src: Input and output tensor for the add_scalar operation
 * */
template<typename T>
void add_scalar(T alpha, const Tensor<T> &src)
{
    add_scalar_async<T>(alpha, src);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void add_scalar_async<fp32_t>(fp32_t alpha, const Tensor<fp32_t> &src);

template
void add_scalar_async<fp64_t>(fp64_t alpha, const Tensor<fp64_t> &src);

// Explicit instantiation
template
void add_scalar<fp32_t>(fp32_t alpha, const Tensor<fp32_t> &src);

template
void add_scalar<fp64_t>(fp64_t alpha, const Tensor<fp64_t> &src);

} // namespace tensor
} // namespace nntile

