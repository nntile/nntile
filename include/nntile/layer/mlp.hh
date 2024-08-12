/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/layer/mlp.hh
 * Multilayer perceptron
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor.hh>
#include <nntile/layer/linear.hh>
#include <nntile/layer/gelutanh.hh>

namespace nntile::layer
{

//! MLP layer
template<typename T>
class MLP
{
    Linear<T> linear1, linear2;
    GeLUTanh<T> gelu;
    tensor::Tensor<T> linear1_out, gelu_out, dldx_linear2_out, dldx_gelu_out;
public:
    MLP(const tensor::TensorTraits &traits1,
            const std::vector<int> &distr1,
            const tensor::TensorTraits &traits2,
            const std::vector<int> &distr2,
            const tensor::TensorTraits &traits_tmp,
            const std::vector<int> &distr_tmp,
            starpu_mpi_tag_t &last_tag):
        linear1(traits1, distr1, last_tag),
        linear2(traits2, distr2, last_tag),
        linear1_out(traits_tmp, distr_tmp, last_tag),
        gelu_out(traits_tmp, distr_tmp, last_tag),
        dldx_linear2_out(traits_tmp, distr_tmp, last_tag),
        dldx_gelu_out(traits_tmp, distr_tmp, last_tag)
    {
    }
    const Linear<T> &get_linear1() const
    {
        return linear1;
    }
    const Linear<T> &get_linear2() const
    {
        return linear2;
    }
    const tensor::Tensor<T> &get_gelu() const
    {
        return gelu_out;
    }
    void forward_async(const tensor::Tensor<T> &input,
            const tensor::Tensor<T> &output) const
    {
        linear1.forward_async(input, linear1_out);
        input.wont_use();
        gelu.forward_async(linear1_out, gelu_out);
        linear2.forward_async(gelu_out, output);
        gelu_out.wont_use();
    }
    void forward_async(const tensor::Tensor<T> &input, T beta,
            const tensor::Tensor<T> &output) const
    {
        linear1.forward_async(input, linear1_out);
        input.wont_use();
        gelu.forward_async(linear1_out, gelu_out);
        linear2.forward_async(T{1}, gelu_out, beta, output);
        gelu_out.wont_use();
    }
    void backward_async(const tensor::Tensor<T> &input,
            const tensor::Tensor<T> &dldx_input,
            const tensor::Tensor<T> &dldx_output,
            const tensor::Tensor<T> &grad_linear1,
            const tensor::Tensor<T> &grad_linear2) const
    {
        linear2.backward_async(gelu_out, dldx_input, dldx_linear2_out,
                grad_linear2);
        gelu_out.invalidate_submit();
        dldx_input.invalidate_submit();
        gelu.backward_async(linear1_out, dldx_linear2_out, dldx_gelu_out);
        linear1_out.invalidate_submit();
        dldx_linear2_out.invalidate_submit();
        linear1.backward_async(input, dldx_gelu_out, dldx_output,
                grad_linear2);
        input.invalidate_submit();
        dldx_gelu_out.invalidate_submit();
    }
    void unregister()
    {
        linear1.unregister();
        linear2.unregister();
        linear1_out.unregister();
        gelu_out.unregister();
        dldx_linear2_out.unregister();
        dldx_gelu_out.unregister();
    }
};

// Explicit instantiations
extern template
class MLP<fp32_t>;

extern template
class MLP<fp64_t>;

} // namespace nntile::layer
