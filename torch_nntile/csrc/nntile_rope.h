/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_rope.h
 */

#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/autograd/custom_function.h>

#include <array>

namespace torch_nntile
{

at::Tensor rope_forward(
    const at::Tensor &sin,
    const at::Tensor &cos,
    const at::Tensor &x);

at::Tensor rope_backward(
    const at::Tensor &sin,
    const at::Tensor &cos,
    const at::Tensor &grad_out,
    std::array<bool, 1> output_mask);

namespace detail
{

class RopeFn : public torch::autograd::Function<RopeFn>
{
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        at::Tensor sin,
        at::Tensor cos,
        at::Tensor x)
    {
        ctx->save_for_backward({sin, cos});
        return rope_forward(sin, cos, x);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        std::array<bool, 1> const mask = {ctx->needs_input_grad(2)};
        auto grad_x = rope_backward(
            saved[0],
            saved[1],
            grad_outputs[0],
            mask);
        torch::autograd::variable_list result(3);
        result[2] = grad_x;
        return result;
    }
};

} // namespace detail

//! Differentiable RoPE (Python ``torch_nntile.rope``).
inline at::Tensor rope(
    const at::Tensor &sin,
    const at::Tensor &cos,
    const at::Tensor &x)
{
    return detail::RopeFn::apply(sin, cos, x);
}

} // namespace torch_nntile
