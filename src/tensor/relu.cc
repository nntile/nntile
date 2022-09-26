/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/relu.cc
 * ReLU operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-26
 * */

#include "nntile/tensor/relu.hh"
#include "nntile/starpu/relu.hh"

namespace nntile
{
namespace tensor
{

//! Asynchronous tensor-wise relu operation
//
// @param[inout] A: Tensor for the element-wise relu operation
template<typename T>
void relu_async(const Tensor<T> &A)
{
    int mpi_rank = starpu_mpi_world_rank();
    for(Index i = 0; i < A.grid.nelems; ++i)
    {
        auto tile_handle = A.get_tile_handle(i);
        int tile_rank = starpu_mpi_data_get_rank(tile_handle);
        // Execute only on node-owner
        if(mpi_rank == tile_rank)
        {
            auto tile_traits = A.get_tile_traits(i);
            starpu::relu::submit<T>(tile_traits.nelems, tile_handle);
        }
        // Flush cache for the output tile on every node
        starpu_mpi_cache_flush(MPI_COMM_WORLD, tile_handle);
    }
}

//! Blocking version of tensor-wise relu operation
//
// @param[inout] A: Tensor for the element-wise relu operation
template<typename T>
void relu(const Tensor<T> &A)
{
    relu_async<T>(A);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void relu<fp32_t>(const Tensor<fp32_t> &A);

template
void relu<fp64_t>(const Tensor<fp64_t> &A);

} // namespace tensor
} // namespace nntile

