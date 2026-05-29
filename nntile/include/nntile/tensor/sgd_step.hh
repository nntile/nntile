/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/sgd_step.hh
 * Fused SGD with momentum step operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

template<typename T>
void sgd_step_async(Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
    const Tensor<T> &grad, const Tensor<T> &velocity, const Tensor<T> &p);

template<typename T>
void sgd_step(Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov,
    const Tensor<T> &grad, const Tensor<T> &velocity, const Tensor<T> &p);

} // namespace nntile::tensor
