/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/hypot_scalar_inverse.cc
 * hypot_scalar_inverse operation for Tensor<T>'s
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-09-28
 * */

#include "nntile/tensor/hypot_scalar_inverse.hh"
#include "nntile/starpu/hypot_scalar_inverse.hh"

namespace nntile
{
namespace tensor
{

template<typename T>
void hypot_scalar_inverse_async(T eps, T alpha, const Tensor<T> &dst)
{
    // Apply per-tile hypot asynchronously as needed
    int mpi_rank = starpu_mpi_world_rank();
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        auto dst_tile_handle = dst.get_tile_handle(i);
        // MPI rank of the destination tile
        int dst_tile_rank = dst_tile_handle.mpi_get_rank();
        // Execute only on destination node
        if(mpi_rank == dst_tile_rank)
        {
            auto traits = dst.get_tile_traits(i);
            starpu::hypot_scalar_inverse::submit<T>(traits.nelems, eps, alpha,
                    dst_tile_handle);
        }
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

template<typename T>
void hypot_scalar_inverse(T eps, T alpha, const Tensor<T> &dst)
{
    hypot_scalar_inverse_async<T>(eps, alpha, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation of template
template
void hypot_scalar_inverse_async<fp32_t>(fp32_t eps, fp32_t alpha,
        const Tensor<fp32_t> &dst);

template
void hypot_scalar_inverse_async<fp64_t>(fp64_t eps, fp64_t alpha,
        const Tensor<fp64_t> &dst);

// Explicit instantiation of template
template
void hypot_scalar_inverse<fp32_t>(fp32_t eps, fp32_t alpha,
        const Tensor<fp32_t> &dst);

template
void hypot_scalar_inverse<fp64_t>(fp64_t eps, fp64_t alpha,
        const Tensor<fp64_t> &dst);

} // namespace tensor
} // namespace nntile
