/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/clear.cc
 * Clear Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-05
 * */

#include "nntile/tile/clear.hh"
#include "nntile/starpu/clear.hh"

namespace nntile
{

template<typename T>
void clear_work(const Tile<T> &src)
{
    nntile::starpu::clear(src);
}

// Explicit instantiation
template
void clear_work<fp32_t>(const Tile<fp32_t> &src);

template
void clear_work<fp64_t>(const Tile<fp64_t> &src);

} // namespace nntile

