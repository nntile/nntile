/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_add_fiber.h
 */

#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/autograd/custom_function.h>

#include <array>
#include <cstdint>
#include <tuple>

namespace torch_nntile
{

at::Tensor add_fiber_forward(
    const at::Tensor &fiber,
    const at::Tensor &tensor,
    int64_t axis,
    int64_t batch_ndim,
    double alpha = 1.0,
    double beta = 1.0);

std::tuple<at::Tensor, at::Tensor> add_fiber_backward(
    const at::Tensor &grad_out,
    const at::Tensor &fiber,
    const at::Tensor &tensor,
    int64_t axis,
    int64_t batch_ndim,
    std::array<bool, 2> output_mask,
    double alpha = 1.0,
    double beta = 1.0);

namespace detail
{

class AddFiberFn : public torch::autograd::Function<AddFiberFn>
{
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        at::Tensor fiber,
        at::Tensor tensor,
        int64_t axis,
        int64_t batch_ndim,
        double alpha,
        double beta)
    {
        ctx->saved_data["axis"] = axis;
        ctx->saved_data["batch_ndim"] = batch_ndim;
        ctx->saved_data["alpha"] = alpha;
        ctx->saved_data["beta"] = beta;
        ctx->save_for_backward({fiber, tensor});
        return add_fiber_forward(
            fiber,
            tensor,
            axis,
            batch_ndim,
            alpha,
            beta);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        int64_t const axis = ctx->saved_data["axis"].toInt();
        int64_t const batch_ndim =
            ctx->saved_data["batch_ndim"].toInt();
        double const alpha = ctx->saved_data["alpha"].toDouble();
        double const beta = ctx->saved_data["beta"].toDouble();
        std::array<bool, 2> const mask = {
            ctx->needs_input_grad(0),
            ctx->needs_input_grad(1),
        };
        auto gb = add_fiber_backward(
            grad_outputs[0],
            saved[0],
            saved[1],
            axis,
            batch_ndim,
            mask,
            alpha,
            beta);
        torch::autograd::variable_list result(6);
        result[0] = std::get<0>(gb);
        result[1] = std::get<1>(gb);
        return result;
    }
};

} // namespace detail

//! Differentiable ``add_fiber`` (Python ``torch_nntile.add_fiber``).
inline at::Tensor add_fiber(
    const at::Tensor &fiber,
    const at::Tensor &tensor,
    int64_t axis,
    int64_t batch_ndim,
    double alpha = 1.0,
    double beta = 1.0)
{
    return detail::AddFiberFn::apply(
        fiber,
        tensor,
        axis,
        batch_ndim,
        alpha,
        beta);
}

} // namespace torch_nntile
