/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_model_transpose.h
 * Cyclic ``tensor::transpose`` for native NNTile-layout models.
 *
 * Native C++ models (GPT-2, BERT, Llama, ...) must use ``model_transpose``
 * here. Do **not** use ``swap_two_axes`` / ``aten::transpose`` - that path
 * exists only for HuggingFace-layout compatibility and is much slower.
 */

#pragma once

#include "nntile_executor.h"

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
        // Y = X.T() => dX = dY.T(); the payload of X is unused. Saving a
        // non-leaf would pin the pre-transpose activation (Q/K/V in
        // model layout, SDPA context in kernel layout) until backward.
        // Keep a leaf only so fused SGD can bind param.grad.
        if (x.is_leaf() && x.requires_grad())
        {
            ctx->save_for_backward({x});
        }
        return model_transpose_forward(x, model_ndim);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        int64_t const model_ndim =
            ctx->saved_data["model_ndim"].toInt();
        at::Tensor saved_x;
        if (!saved.empty())
        {
            saved_x = saved[0];
        }
        auto grad = model_transpose_backward(
            grad_outputs[0],
            model_ndim,
            saved_x);
        torch::autograd::variable_list result(2);
        result[0] = grad;
        return result;
    }
};

} // namespace detail

//! Differentiable cyclic model-code transpose (NNTile SDPA layout).
inline at::Tensor model_transpose(
    const at::Tensor &x,
    int64_t model_ndim)
{
    return detail::ModelTransposeFn::apply(x, model_ndim);
}

} // namespace torch_nntile
