/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_transpose.h
 */

#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/autograd/custom_function.h>

#include <cstdint>

namespace torch_nntile
{

at::Tensor model_transpose_forward(
    const at::Tensor &x,
    int64_t model_ndim);

at::Tensor model_transpose_backward(
    const at::Tensor &grad_out,
    int64_t model_ndim,
    const at::Tensor &x = {});

namespace detail
{

class ModelTransposeFn :
    public torch::autograd::Function<ModelTransposeFn>
{
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        at::Tensor x,
        int64_t model_ndim)
    {
        ctx->saved_data["model_ndim"] = model_ndim;
        ctx->save_for_backward({x});
        return model_transpose_forward(x, model_ndim);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        int64_t const model_ndim =
            ctx->saved_data["model_ndim"].toInt();
        auto grad = model_transpose_backward(
            grad_outputs[0],
            model_ndim,
            saved[0]);
        torch::autograd::variable_list result(2);
        result[0] = grad;
        return result;
    }
};

} // namespace detail

//! Differentiable model-code transpose (Python ``nntile_model_transpose``).
inline at::Tensor model_transpose(
    const at::Tensor &x,
    int64_t model_ndim)
{
    return detail::ModelTransposeFn::apply(x, model_ndim);
}

} // namespace torch_nntile
