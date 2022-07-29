/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/scal.hh
 * Scaling operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once
#include <nntile/tensor/tensor.hh>
#include "nntile/tile/scal.hh"

namespace nntile
{

template<typename T>
void scal_work(const Tensor<T> &src, T alpha);

} // namespace nntile

