/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/sgd_momentum.cc
 * SGD_MOMENTUM operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/sgd_momentum.hh"
#include "nntile/starpu/sgd_momentum.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise sgd_momentum operation
//
// @param[inout] A: Tensor for the element-wise sgd_momentum operation
template<typename T>
void sgd_momentum_async(const Tensor<T> &A)
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
            starpu::sgd_momentum::submit<T>(tile_traits.nelems, tile_handle);
        }
        // Flush cache for the output tile on every node
        tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise sgd_momentum operation
//
// @param[inout] A: Tensor for the element-wise sgd_momentum operation
template<typename T>
void sgd_momentum(const Tensor<T> &A)
{
    sgd_momentum_async<T>(A);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void sgd_momentum_async<fp32_t>(const Tensor<fp32_t> &A);

template
void sgd_momentum_async<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &A);

template
void sgd_momentum_async<fp64_t>(const Tensor<fp64_t> &A);

// Explicit instantiation
template
void sgd_momentum<fp32_t>(const Tensor<fp32_t> &A);

template
void sgd_momentum<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &A);

template
void sgd_momentum<fp64_t>(const Tensor<fp64_t> &A);

} // namespace nntile::tensor
