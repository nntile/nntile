/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/layer/linear.hh
 * Fully connected dense linear layer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-11-07
 * */

#pragma once

#include <nntile/layer/base.hh>

namespace nntile
{
namespace layer
{

//! Linear layer
template<typename T>
class Linear: public Base<T>
{
    tensor::Tensor<T> weight, grad_weight;
public:
    Linear(const tensor::TensorTraits &traits,
            const std::vector<int> &distribution,
            starpu_mpi_tag_t &last_tag):
        weight(traits, distribution, last_tag),
        grad_weight(traits, distribution, last_tag)
    {
    }
    ~Linear()
    {
        weight.unregister();
        grad_weight.unregister();
    }
    void forward_async(const tensor::Tensor<T> &input,
            const tensor::Tensor<T> &output) const
    {
        constexpr T one = 1, zero = 0;
        constexpr TransOp opN(TransOp::NoTrans);
        tensor::gemm_async<T>(one, opN, weight, opN, input, zero, output, 1);
        input.wont_use();
        weight.wont_use();
    }
    void backward_async(const tensor::Tensor<T> &forward_input,
            const tensor::Tensor<T> &input,
            const tensor::Tensor<T> &output) const
    {
        constexpr T one = 1, zero = 0;
        constexpr TransOp opN(TransOp::NoTrans), opT(TransOp::Trans);
        tensor::gemm_async<T>(one, opN, input, opT, forward_input, zero,
                grad_weight, 1);
        forward_input.invalidate_submit();
        tensor::gemm_async<T>(one, opT, weight, opN, input, zero, output, 1);
        weight.wont_use();
        input.invalidate_submit();
    }
    void grad_descent(T rate) const
    {
        tensor::axpy<T>(rate, grad_weight, weight);
        grad_weight.invalidate_submit();
    }
};

// Explicit instantiations
extern template
class Linear<fp32_t>;

extern template
class Linear<fp64_t>;

} // namespace layer
} // namespace nntile

