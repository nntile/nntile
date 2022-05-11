/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/relu.hh
 * ReLU operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{

//! Asynchronous tensor-wise ReLU operation
//
// @param[inout] A: Tensor for the element-wise ReLU operation
template<typename T>
void relu_async(const Tensor<T> &A);

extern template
void relu_async(const Tensor<fp32_t> &A);

extern template
void relu_async(const Tensor<fp64_t> &A);

//! Blocking version of tensor-wise ReLU operation
//
// @param[inout] A: Tensor for the element-wise ReLU operation
template<typename T>
void relu(const Tensor<T> &A)
{
    relu_async<T>(A);
    starpu_task_wait_for_all();
}

} // namespace nntile

