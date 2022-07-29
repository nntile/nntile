/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/sumnorm.hh
 * Sum and Euclidian norm of Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{

template<typename T>
void sumnorm_async(const Tensor<T> &src, const Tensor<T> &sumnorm, Index axis);

template<typename T>
void sumnorm(const Tensor<T> &src, const Tensor<T> &sumnorm, Index axis)
{
    sumnorm_async(src, sumnorm, axis);
    starpu_task_wait_for_all();
}

} // namespace nntile

