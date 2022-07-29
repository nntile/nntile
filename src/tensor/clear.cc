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
 * @date 2022-04-22
 * */

#include "nntile/tensor/clear.hh"
#include "nntile/tile/clear.hh"

namespace nntile
{

template<typename T>
void clear_work(const Tensor<T> &src)
{
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        clear_work<T>(src.get_tile(i));
    }
}

// Explicit instantiation
template
void clear_work<fp32_t>(const Tensor<fp32_t> &src);

template
void clear_work<fp64_t>(const Tensor<fp64_t> &src);

} // namespace nntile

