/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is a software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the StarPU runtime system.
 *
 * @file include/nntile/tensor/lion_step.hh
 * Fused Lion step operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

/*!
 * Asynchronous tensor-wise fused Lion step.
 *
 * @param[in] num_iter: Current iteration number.
 * @param[in] beta_1: Momentum coefficient for the exponential moving average of gradients.
 * @param[in] beta_2: Momentum coefficient for the exponential moving average update (unused in Lion, but kept for API symmetry if needed).
 * @param[in] lambda: Penalty coefficient applied to the parameters.
 * @param[in] lr: Learning rate.
 * @param[in] weight_decay: L2 regularization coefficient.
 * @param[in] grad: Tensor containing gradients.
 * @param[in] first_moment: Tensor containing the momentum (first moment).
 * @param[in] p: Tensor containing the model parameters.
 */
template<typename T>
void lion_step_async(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar lambda, Scalar lr, Scalar weight_decay,
                     const Tensor<T> &grad, const Tensor<T> &first_moment,
                     const Tensor<T> &p);

/*!
 * Blocking version of the tensor-wise fused Lion step.
 *
 * @param[in] num_iter: Current iteration number.
 * @param[in] beta_1: Momentum coefficient for the exponential moving average of gradients.
 * @param[in] beta_2: Momentum coefficient for the exponential moving average update.
 * @param[in] lambda: Penalty coefficient applied to the parameters.
 * @param[in] lr: Learning rate.
 * @param[in] weight_decay: L2 regularization coefficient.
 * @param[in] grad: Tensor containing gradients.
 * @param[in] first_moment: Tensor containing the momentum (first moment).
 * @param[in] p: Tensor containing the model parameters.
 */
template<typename T>
void lion_step(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar lambda, Scalar lr, Scalar weight_decay,
               const Tensor<T> &grad, const Tensor<T> &first_moment,
               const Tensor<T> &p);

} // namespace nntile::tensor
