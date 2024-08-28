/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/scal_inplace.cc
 * Inplace scal of Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/scal_inplace.hh"
#include "nntile/starpu/scal_inplace.hh"

namespace nntile::tensor
{

//! scal_inplacee tensor
template<typename T>
void scal_inplace_async(Scalar alpha, const Tensor<T> &data)
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
            starpu::scal_inplace::submit<T>(data_tile_traits.nelems, alpha,
                    data_tile_handle);
        }
        // Flush cache for the output tile on every node
        data_tile_handle.mpi_flush();
    }
}

template<typename T>
void scal_inplace(Scalar alpha, const Tensor<T> &data)
{
    scal_inplace_async<T>(alpha, data);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void scal_inplace_async<fp32_t>(Scalar alpha, const Tensor<fp32_t> &data);

template
void scal_inplace_async<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &data);

template
void scal_inplace_async<fp64_t>(Scalar alpha, const Tensor<fp64_t> &data);

// Explicit instantiation
template
void scal_inplace<fp32_t>(Scalar alpha, const Tensor<fp32_t> &data);

template
void scal_inplace<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &data);

template
void scal_inplace<fp64_t>(Scalar alpha, const Tensor<fp64_t> &data);

} // namespace nntile::tensor
