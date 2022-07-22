/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/bias2.hh
 * Bias operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{

//! Tensor-wise bias operation by averages and deviations
//
// Main computational routine that does NO argument checking.
// @param[in] avg_dev: Source of the bias (averages and deviations)
// @param[inout] dst: Destination of the bias
// @param[in] axis: Dimension index of the bias
template<typename T>
void bias2_work(const Tensor<T> &avg_dev, const Tensor<T> &dst,
        Index axis);

//! Tensor-wise bias operation by averages and deviations
//
// Checks input arguments
template<typename T>
void bias2_async(const Tensor<T> &avg_dev, const Tensor<T> &dst,
        Index axis)
{
    // Check dimensions
    if(dst.ndim != avg_dev.ndim)
    {
        throw std::runtime_error("dst.ndim != avg_dev.ndim");
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
    if(avg_dev.shape[0] != 2)
    {
        throw std::runtime_error("avg_dev.shape[0] != 2");
    }
    if(avg_dev.basetile_shape[0] != 2)
    {
        throw std::runtime_error("avg_dev.basetile_shape[0] != 2");
    }
    for(Index i = 0; i < axis; ++i)
    {
        if(dst.shape[i] != avg_dev.shape[i+1])
        {
            throw std::runtime_error("dst.shape[i] != avg_dev.shape[i+1]");
        }
        if(dst.basetile_shape[i] != avg_dev.basetile_shape[i+1])
        {
            throw std::runtime_error("dst.basetile_shape[i] != "
                    "avg_dev.basetile_shape[i+1]");
        }
    }
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != avg_dev.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != avg_dev.shape[i]");
        }
        if(dst.basetile_shape[i] != avg_dev.basetile_shape[i])
        {
            throw std::runtime_error("dst.basetile_shape[i] != "
                    "avg_dev.basetile_shape[i]");
        }
    }
    // Launch codelets
    bias2_work<T>(avg_dev, dst, axis);
}

//! Tensor-wise bias operation by averages and deviations
//
// Checks input arguments and blocks until finished
template<typename T>
void bias2(const Tensor<T> &avg_dev, const Tensor<T> &dst, Index axis)
{
    bias2_async<T>(avg_dev, dst, axis);
    starpu_task_wait_for_all();
}

} // namespace nntile

