/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/base.hh
 * Base neural network model as a chain of layers
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-11-17
 * */

#pragma once

#include <nntile/layer/base.hh>

namespace nntile
{
namespace model
{

//! Common API for all models
template<typename T>
class Base
{
    const tensor::TensorTraits input_traits, output_traits;
    std::vector<std::shared_ptr<const layer::Base<T>>> layers;
public:
    Base(const std::vector<std::shared_ptr<const layer::Base<T>>> &layers_):
        input_traits(layers_[0].get()[0].get_input_traits()),
        output_traits(layers_[layers_.size()-1].get()[0].get_output_traits()),
        layers(layers_)
    {
    }
    // Destructor is virtual since this is a base class for all layers
    virtual ~Base() = default;
    void forward_async(const tensor::Tensor<T> &input,
            const tensor::Tensor<T> &output) const = delete;
    void backward_async(const tensor::Tensor<T> &forward_input,
            const tensor::Tensor<T> &input) const = delete;
    void grad_descent(T rate) const = delete;
    const tensor::TensorTraits &get_input_traits() const
    {
        return input_traits;
    }
    const tensor::TensorTraits &get_output_traits() const
    {
        return output_traits;
    }
};

} // namespace model
} // namespace nntile

