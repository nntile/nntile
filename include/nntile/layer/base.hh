/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/layer/base.hh
 * Base API for all layers
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-11-07
 * */

#pragma once

#include <nntile/tensor.hh>

namespace nntile
{
namespace layer
{

// Common API for all layers
template<typename T>
class Base
{
public:
    virtual ~Base() = default;
    void forward_async(const tensor::Tensor<T> &input,
            const tensor::Tensor<T> &output) const = delete;
    void backward_async(const tensor::Tensor<T> &forward_input,
            const tensor::Tensor<T> &input,
            const tensor::Tensor<T> &output) const = delete;
    void grad_descent(T rate) const = delete;
};

} // namespace layer
} // namespace nntile

