/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/layer/linear.hh
 * Fully connected dense linear layer
 *
 * @version 1.0.0
 * */

#pragma once

#include <nntile/layer/base.hh>
#include <nntile/tensor/gemm.hh>
#include <nntile/tensor/strassen.hh>

namespace nntile::layer
{

//! Linear layer
template<typename T>
class Linear: public Base<T>
{
public:
    tensor::Tensor<T> &weight;
    tensor::Tensor<T> &grad_weight;
    Linear(const tensor::TensorTraits &input_traits_,
            const tensor::TensorTraits &output_traits_,
            tensor::Tensor<T> params_,
            tensor::Tensor<T> grads_,
			bool strassen_ = false):
        Base<T>(input_traits_, output_traits_, {params_}, {grads_}),
        weight(this->params[0]),
        grad_weight(this->grads[0])
    {
        if(strassen_)
            p_tensor_gemm = &(tensor::strassen_async<T>);
        else
            p_tensor_gemm = &(tensor::gemm_async<T>);
    }
    virtual ~Linear() = default;
    virtual void forward_async(const tensor::Tensor<T> &input,
            const tensor::Tensor<T> &output) const
    {
        constexpr T one = 1, zero = 0;
        constexpr TransOp opN(TransOp::NoTrans);
        p_tensor_gemm(one, opN, weight, opN, input, zero, output, 1);
        input.wont_use();
        weight.wont_use();
    }
    virtual void backward_async(const tensor::Tensor<T> &forward_input,
            const tensor::Tensor<T> &input,
            const tensor::Tensor<T> &output) const
    {
        constexpr T one = 1, zero = 0;
        constexpr TransOp opN(TransOp::NoTrans), opT(TransOp::Trans);
        p_tensor_gemm(one, opN, input, opT, forward_input, zero, grad_weight,
                      1);
        forward_input.invalidate_submit();
        p_tensor_gemm(one, opT, weight, opN, input, zero, output, 1);
        weight.wont_use();
        input.invalidate_submit();
    }

private:
    template <typename T>
    void (*p_tensor_gemm)(Scalar alpha, const TransOp &transA,
                          const Tensor<T> &A, const TransOp &transB,
                          const Tensor<T> &B, Scalar beta, const Tensor<T> &C,
                          Index ndim, Index batch_ndim,
                          int redux = 0) = nullptr;
};

// Explicit instantiations
extern template
class Linear<fp32_t>;

extern template
class Linear<fp64_t>;

} // namespace nntile::layer
