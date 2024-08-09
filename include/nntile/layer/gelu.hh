/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/layer/gelu.hh
 * GeLU layer
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/layer/base.hh>
#include <nntile/tensor.hh>

namespace nntile::layer
{

//! GeLU layer
template<typename T>
class GeLU: public Base<T>
{
public:
    virtual void forward_async(const tensor::Tensor<T> &input,
            const tensor::Tensor<T> &output) const
    {
        tensor::copy_async<T>(input, output);
        input.wont_use();
        tensor::gelu_async<T>(output);
    }
    virtual void backward_async(const tensor::Tensor<T> &input,
            const tensor::Tensor<T> &dldx_input,
            const tensor::Tensor<T> &dldx_output) const
    {
        tensor::copy_async(input, dldx_output);
        input.invalidate_submit();
        tensor::dgelu_async(dldx_output);
        tensor::prod_async(dldx_input, dldx_output);
        dldx_input.invalidate_submit();
    }
};

// Explicit instantiations
extern template
class GeLU<fp32_t>;

extern template
class GeLU<fp64_t>;

} // namespace nntile::layer
