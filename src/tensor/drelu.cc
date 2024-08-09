/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/drelu.cc
 * Derivative of ReLU operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/drelu.hh"
#include "nntile/starpu/drelu.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise derivative of ReLU operation
//
// @param[inout] A: Tensor for the element-wise derivative of ReLU operation
template<typename T>
void drelu_async(const Tensor<T> &A)
{
    int mpi_rank = starpu_mpi_world_rank();
    for(Index i = 0; i < A.grid.nelems; ++i)
    {
        auto tile_handle = A.get_tile_handle(i);
        int tile_rank = tile_handle.mpi_get_rank();
        // Execute only on node-owner
        if(mpi_rank == tile_rank)
        {
            auto tile_traits = A.get_tile_traits(i);
            starpu::drelu::submit<T>(tile_traits.nelems, tile_handle);
        }
        // Flush cache for the output tile on every node
        tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise derivative of ReLU operation
//
// @param[inout] A: Tensor for the element-wise derivative of ReLU operation
template<typename T>
void drelu(const Tensor<T> &A)
{
    drelu_async<T>(A);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void drelu_async<fp32_t>(const Tensor<fp32_t> &A);

template
void drelu_async<fp64_t>(const Tensor<fp64_t> &A);

// Explicit instantiation
template
void drelu<fp32_t>(const Tensor<fp32_t> &A);

template
void drelu<fp64_t>(const Tensor<fp64_t> &A);

} // namespace nntile::tensor
