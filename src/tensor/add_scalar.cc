/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/add_scalar.cc
 * Add_scalar operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/add_scalar.hh"
#include "nntile/starpu/add_scalar.hh"

namespace nntile::tensor
{

//! Tensor-wise add_scalar operation
template<typename T>
void add_scalar_async(Scalar alpha, Scalar beta, const Tensor<T> &dst)
{
    // Do nothing if alpha is zero
    if(alpha == 0.0 && beta == 1.)
    {
        return;
    }
    // Apply per-tile add_scalar asynchronously as needed
    int mpi_rank = starpu_mpi_world_rank();
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        // Get handle for corresponding tiles of src and dst
        auto dst_tile_handle = dst.get_tile_handle(i);
        // MPI rank of the destination tile
        int dst_tile_rank = dst_tile_handle.mpi_get_rank();
        // Execute only on destination node
        if(mpi_rank == dst_tile_rank)
        {
            auto traits = dst.get_tile_traits(i);
            starpu::add_scalar::submit<T>(traits.nelems, alpha, beta,
                    dst_tile_handle);
        }
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

//! Tensor-wise add_scalar operation
template<typename T>
void add_scalar(Scalar alpha, Scalar beta, const Tensor<T> &dst)
{
    add_scalar_async<T>(alpha, beta, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation of template
template
void add_scalar_async<fp32_t>(Scalar alpha, Scalar beta,
        const Tensor<fp32_t> &dst);

template
void add_scalar_async<fp64_t>(Scalar alpha, Scalar beta,
        const Tensor<fp64_t> &dst);

// Explicit instantiation of template
template
void add_scalar<fp32_t>(Scalar alpha, Scalar beta,
        const Tensor<fp32_t> &dst);

template
void add_scalar<fp64_t>(Scalar alpha, Scalar beta,
        const Tensor<fp64_t> &dst);

} // namespace nntile::tensor
