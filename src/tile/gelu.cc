/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/gelu.cc
 * GeLU operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-08
 * */

#include "nntile/tile/gelu.hh"
#include "nntile/starpu/gelu.hh"

namespace nntile
{

template<typename T>
void gelu_work(const Tile<T> &A)
{
    nntile::starpu::gelu<T>(A.nelems, A);
}

template
void gelu_work<fp32_t>(const Tile<fp32_t> &A);

template
void gelu_work<fp64_t>(const Tile<fp64_t> &A);

} // namespace nntile

