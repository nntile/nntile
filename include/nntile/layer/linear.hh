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
 * @date 2022-11-03
 * */

#pragma once

#include <nntile/tensor.hh>

namespace nntile
{
namespace layer
{

//! Linear layer
template<typename T>
class Linear
{
    tensor::Tensor<T> weight;
public:
    Linear(const tensor::TensorTraits &traits,
            const std::vector<int> &distribution,
            starpu_mpi_tag_t &last_tag):
        weight(traits, distribution, last_tag)
    {
    }
    const tensor::Tensor<T> &get_weight() const
    {
        return weight;
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
    void forward_async(T alpha, const tensor::Tensor<T> &input, T beta,
            const tensor::Tensor<T> &output) const
    {
        constexpr TransOp opN(TransOp::NoTrans);
        tensor::gemm_async<T>(alpha, opN, weight, opN, input, beta, output, 1);
        input.wont_use();
        weight.wont_use();
    }
    void backward_async(const tensor::Tensor<T> &input,
            const tensor::Tensor<T> &dldx_input,
            const tensor::Tensor<T> &dldx_output,
            const tensor::Tensor<T> &grad_weight) const
    {
        constexpr T one = 1, zero = 0;
        constexpr TransOp opN(TransOp::NoTrans), opT(TransOp::Trans);
        tensor::gemm_async<T>(one, opN, dldx_input, opT, input, zero,
                grad_weight, 1);
        input.wont_use();
        tensor::gemm_async<T>(one, opT, weight, opN, dldx_input, zero,
                dldx_output, 1);
        weight.wont_use();
        dldx_input.invalidate_submit();
    }
    void unregister()
    {
        weight.unregister();
    }
};

// Explicit instantiations
extern template
class Linear<fp32_t>;

extern template
class Linear<fp64_t>;

} // namespace layer
} // namespace nntile

