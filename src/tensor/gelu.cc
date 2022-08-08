/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/gelu.cc
 * GeLU operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-08
 * */

#include "nntile/tensor/gelu.hh"
#include "nntile/starpu/gelu.hh"

namespace nntile
{

template<typename T>
void gelu_work(const Tensor<T> &A)
{
    for(Index i = 0; i < A.grid.nelems; ++i)
    {
        auto &tile = A.get_tile(i);
        nntile::starpu::gelu<T>(tile.nelems, tile);
    }
}

template
void gelu_work(const Tensor<fp32_t> &A);

template
void gelu_work(const Tensor<fp64_t> &A);

} // namespace nntile

