/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/gelu.hh
 * GeLU operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{

//! Asynchronous tensor-wise GeLU operation
//
// @param[inout] A: Tensor for the element-wise GeLU operation
template<typename T>
void gelu_async(const Tensor<T> &A);

extern template
void gelu_async(const Tensor<fp32_t> &A);

extern template
void gelu_async(const Tensor<fp64_t> &A);

//! Blocking version of tensor-wise GeLU operation
//
// @param[inout] A: Tensor for the element-wise GeLU operation
template<typename T>
void gelu(const Tensor<T> &A)
{
    gelu_async<T>(A);
    starpu_task_wait_for_all();
}

} // namespace nntile

