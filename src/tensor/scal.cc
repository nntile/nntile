/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/scal.cc
 * Scaling operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tensor/scal.hh"
#include "nntile/tile/scal.hh"

namespace nntile
{

template<typename T>
void scal_work(const Tensor<T> &src, T alpha)
{
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        scal_work<T>(src.get_tile(i), alpha);
    }
}

template
void scal_work<fp32_t>(const Tensor<fp32_t> &src, fp32_t alpha);

template
void scal_work<fp64_t>(const Tensor<fp64_t> &src, fp64_t alpha);

} // namespace nntile

