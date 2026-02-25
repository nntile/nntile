/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/sqrt.cc
 * Sqrt operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/sqrt.hh"
#include "nntile/starpu/sqrt.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tile
{

//! Asynchronous tile-wise sqrt operation
/*! @param[inout] A: Tile for the element-wise sqrt operation
 * */
template<typename T>
void sqrt_async(const Tile<T> &src, const Tile<T> &dst)
{
    int mpi_rank = starpu_mpi_world_rank();
    int dst_rank = dst.mpi_get_rank();
    src.mpi_transfer(dst_rank, mpi_rank);
    if(mpi_rank == dst_rank)
    {
        // Submit task without any arguments checked
        starpu::sqrt.submit<std::tuple<T>>(src.nelems, src, dst);
    }
}

//! Blocking version of tile-wise sqrt operation
/*! @param[inout] A: Tile for the element-wise sqrt operation
 * */
template<typename T>
void sqrt(const Tile<T> &src, const Tile<T> &dst)
{
    sqrt_async<T>(src, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void sqrt_async<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void sqrt_async<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

// Explicit instantiation
template
void sqrt<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void sqrt<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

} // namespace nntile::tile
