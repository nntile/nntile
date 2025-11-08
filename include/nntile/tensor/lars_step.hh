/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/lars_step.hh
 * Fuse LARS step operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

template<typename T>
void lars_step_async(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
    const Tensor<T> &grad, const Tensor<T> &p,
    const Tensor<fp32_t> &grad_norm, const Tensor<fp32_t> &p_norm);

template<typename T>
void lars_step(Scalar lr, Scalar trust_ratio, Scalar weight_decay,
    const Tensor<T> &grad, const Tensor<T> &p,
    const Tensor<fp32_t> &grad_norm, const Tensor<fp32_t> &p_norm);

} // namespace nntile::tensor
