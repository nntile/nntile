/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/bias.hh
 * Bias operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-05
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{

//! Tensor-wise bias operation
//
// Main computational routine that does NO argument checking.
// The source tensor shall have 1 dimension less than the destination tensor,
// as this operation does the following update:
// dst[i_0, ..., i_b-1, i_b, i_b+1, ..., i_d-1] += src[i_0, ..., i_b-1, i_b+1,
// ..., i_d-1]
// where b is the axis and i_d is the src.ndim
//
// @param[in] src: Source of the bias
// @param[inout] dst: Destination of the bias
// @param[in] axis: Dimension index of the bias
template<typename T>
void bias_work(const Tensor<T> &src, const Tensor<T> &dst, Index axis);

//! Tensor-wise bias operation
//
// Checks input arguments
template<typename T>
void bias_async(const Tensor<T> &src, const Tensor<T> &dst, Index axis)
{
    // Check dimensions
    if(dst.ndim != src.ndim+1)
    {
        throw std::runtime_error("dst.ndim != src.ndim+1");
    }
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= dst.ndim)
    {
        throw std::runtime_error("axis >= dst.ndim");
    }
    // Check shapes of tensors
    for(Index i = 0; i < axis; ++i)
    {
        if(dst.shape[i] != src.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i]");
        }
        if(dst.basetile_shape[i] != src.basetile_shape[i])
        {
            throw std::runtime_error("dst.basetile_shape[i] != "
                    "src.basetile_shape[i]");
        }
    }
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src.shape[i-1])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i-1]");
        }
        if(dst.basetile_shape[i] != src.basetile_shape[i-1])
        {
            throw std::runtime_error("dst.basetile_shape[i] != "
                    "src.basetile_shape[i-1]");
        }
    }
    // Launch codelets
    bias_work<T>(src, dst, axis);
}

//! Tensor-wise bias operation
//
// Checks input arguments and blocks until finished
template<typename T>
void bias(const Tensor<T> &src, const Tensor<T> &dst, Index axis)
{
    bias_async<T>(src, dst, axis);
    starpu_task_wait_for_all();
}

} // namespace nntile

