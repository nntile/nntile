/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/clear.cc
 * Clear Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-10
 * */

#include "nntile/tensor/clear.hh"
#include "nntile/starpu/clear.hh"

namespace nntile
{
namespace tensor
{

template<typename T>
void clear_async(const Tensor<T> &dst)
{
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        starpu::clear::submit(dst.get_tile_handle(i));
    }
}

template<typename T>
void clear(const Tensor<T> &dst)
{
    clear_async<T>(dst);
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void clear<fp32_t>(const Tensor<fp32_t> &dst);

template
void clear<fp64_t>(const Tensor<fp64_t> &dst);

} // namespace tensor
} // namespace nntile

