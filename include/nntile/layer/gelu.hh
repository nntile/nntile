/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/layer/gelu.hh
 * GeLU layer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-10-26
 * */

#pragma once

#include <nntile/tensor.hh>

namespace nntile
{
namespace layer
{

//! GeLU layer
template<typename T>
class GeLU
{
public:
    void forward_async(const tensor::Tensor<T> &input,
            const tensor::Tensor<T> &output) const
    {
        tensor::copy_async<T>(input, output);
        tensor::gelu_async<T>(output);
    }
    void backward_async(const tensor::Tensor<T> &input,
            const tensor::Tensor<T> &dldx_input,
            const tensor::Tensor<T> &dldx_output) const
    {

    }
};

// Explicit instantiations
extern template
class GeLU<fp32_t>;

extern template
class GeLU<fp64_t>;

} // namespace layer
} // namespace nntile

