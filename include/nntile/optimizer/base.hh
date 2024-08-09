/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/optimizer/base.hh
 * Common API for all optimizers
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::optimizer
{

//! Common API for all optimizers
template<typename T>
class Base
{
public:
    std::vector<tensor::Tensor<T>> params;
    std::vector<tensor::Tensor<T>> grads;
    Base(const std::vector<tensor::Tensor<T>> &params_,
            const std::vector<tensor::Tensor<T>> &grads_):
        params(params_),
        grads(grads_)
    {
    }
    virtual void update() = 0;
    virtual ~Base() = default;
};

} // namespace nntile::optimizer
