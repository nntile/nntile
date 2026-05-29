/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/adamw_step.hh
 * Fused AdamW step operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

template<typename T>
void adamw_step_async(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
    const Tensor<T> &grad, const Tensor<T> &first_moment, const Tensor<T> &second_moment,
                   const Tensor<T> &p);

template<typename T>
void adamw_step(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
    const Tensor<T> &grad, const Tensor<T> &first_moment, const Tensor<T> &second_moment,
                   const Tensor<T> &p);

} // namespace nntile::tensor
