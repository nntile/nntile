/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/base.hh
 * Base neural network model as a chain of layers
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/layer/base.hh>

namespace nntile::model
{

//! Common API for all models
template<typename T>
class Base
{
public:
    tensor::TensorTraits input_traits, output_traits;
    std::vector<std::shared_ptr<layer::Base<T>>> layers;
    std::vector<tensor::Tensor<T>> params;
    std::vector<tensor::Tensor<T>> grads;
    std::vector<tensor::Tensor<T>> tmps;
    Base(const std::vector<std::shared_ptr<layer::Base<T>>> &layers_):
        input_traits(layers_[0].get()[0].input_traits),
        output_traits(layers_[layers_.size()-1].get()[0].output_traits),
        layers(layers_),
        params(),
        grads()
    {
        for(auto l: layers)
        {
            params.insert(params.cend(), l.get()->params.cbegin(),
                    l.get()->params.cend());
            grads.insert(grads.cend(), l.get()->grads.cbegin(),
                    l.get()->grads.cend());
        }
    }
    // Destructor is virtual since this is a base class for all layers
    virtual ~Base() = default;
    void forward_async(const tensor::Tensor<T> &input,
            const tensor::Tensor<T> &output) const
    {
    }
    void backward_async(const tensor::Tensor<T> &forward_input,
            const tensor::Tensor<T> &input) const
    {
    }
};

} // namespace nntile::model
