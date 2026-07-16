/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_transpose.h
 */

#pragma once

#include "nntile_executor.h"

#include <ATen/ATen.h>
#include <torch/csrc/autograd/custom_function.h>

#include <cstdint>
#include <vector>

namespace torch_nntile
{

at::Tensor model_transpose_forward(
    const at::Tensor &x,
    int64_t model_ndim);

at::Tensor model_transpose_backward(
    const at::Tensor &grad_out,
    int64_t model_ndim,
    const at::Tensor &x = {});

//! Materializing axis swap (contiguous out).
inline at::Tensor swap_two_axes_forward(
    const at::Tensor &x,
    int64_t dim0,
    int64_t dim1)
{
    int64_t const n = x.dim();
    if (dim0 < 0)
    {
        dim0 += n;
    }
    if (dim1 < 0)
    {
        dim1 += n;
    }
    if (dim0 == dim1)
    {
        return x;
    }
    auto sizes = x.sizes().vec();
    std::swap(
        sizes[static_cast<std::size_t>(dim0)],
        sizes[static_cast<std::size_t>(dim1)]);
    at::Tensor out = at::empty(
        sizes,
        x.options().memory_format(at::MemoryFormat::Contiguous));
    tensor_swap_two_axes_fp32(x, out, dim0, dim1);
    return out;
}

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

class SwapTwoAxesFn :
    public torch::autograd::Function<SwapTwoAxesFn>
{
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        at::Tensor x,
        int64_t dim0,
        int64_t dim1)
    {
        ctx->saved_data["dim0"] = dim0;
        ctx->saved_data["dim1"] = dim1;
        return swap_two_axes_forward(x, dim0, dim1);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        int64_t const dim0 = ctx->saved_data["dim0"].toInt();
        int64_t const dim1 = ctx->saved_data["dim1"].toInt();
        torch::autograd::variable_list result(3);
        result[0] = swap_two_axes_forward(
            grad_outputs[0],
            dim0,
            dim1);
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

//! Differentiable materializing axis swap (avoids as_strided backward).
inline at::Tensor swap_two_axes(
    const at::Tensor &x,
    int64_t dim0,
    int64_t dim1)
{
    return detail::SwapTwoAxesFn::apply(x, dim0, dim1);
}

} // namespace torch_nntile
