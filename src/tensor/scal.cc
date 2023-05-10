/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/scal.cc
 * Euclidean norm of Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-05
 * */

#include "nntile/tensor/scal.hh"
#include "nntile/starpu/scal.hh"

namespace nntile
{
namespace tensor
{

//! Scale tensor
template<typename T>
void scal_async(T alpha, const Tensor<T> &data)
{
    // Do actual calculations
    int mpi_rank = starpu_mpi_world_rank();
    for(Index i = 0; i < data.grid.nelems; ++i)
    {
        auto data_tile_handle = data.get_tile_handle(i);
        auto data_tile_traits = data.get_tile_traits(i);
        int data_tile_rank = data_tile_handle.mpi_get_rank();
        // Execute on source tile
        if(mpi_rank == data_tile_rank)
        {
            starpu::scal::submit<T>(alpha, data_tile_traits.nelems,
                    data_tile_handle);
        }
        // Flush cache for the output tile on every node
        data_tile_handle.mpi_flush();
    }
}

template<typename T>
void scal(T alpha, const Tensor<T> &data)
{
    scal_async<T>(alpha, data);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void scal_async<fp32_t>(fp32_t alpha, const Tensor<fp32_t> &data);

template
void scal_async<fp64_t>(fp64_t alpha, const Tensor<fp64_t> &data);

// Explicit instantiation
template
void scal<fp32_t>(fp32_t alpha, const Tensor<fp32_t> &data);

template
void scal<fp64_t>(fp64_t alpha, const Tensor<fp64_t> &data);

} // namespace tensor
} // namespace nntile

