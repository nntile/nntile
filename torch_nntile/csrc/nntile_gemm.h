/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_gemm.h
 */

#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/autograd/custom_function.h>

#include <array>
#include <cstdint>
#include <tuple>

namespace torch_nntile
{

at::Tensor gemm_forward(
    const at::Tensor &a,
    const at::Tensor &b,
    int64_t ndim,
    int64_t batch_ndim,
    bool trans_a = false,
    bool trans_b = false);

std::tuple<at::Tensor, at::Tensor> gemm_backward(
    const at::Tensor &a,
    const at::Tensor &b,
    const at::Tensor &grad_out,
    int64_t ndim,
    int64_t batch_ndim,
    std::array<bool, 2> output_mask,
    bool trans_a = false,
    bool trans_b = false);

at::Tensor matmul_nd(
    const at::Tensor &a,
    const at::Tensor &b);

namespace detail
{

class GemmFn : public torch::autograd::Function<GemmFn>
{
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        at::Tensor a,
        at::Tensor b,
        int64_t ndim,
        int64_t batch_ndim,
        bool trans_a,
        bool trans_b)
    {
        ctx->saved_data["ndim"] = ndim;
        ctx->saved_data["batch_ndim"] = batch_ndim;
        ctx->saved_data["trans_a"] = trans_a;
        ctx->saved_data["trans_b"] = trans_b;
        ctx->save_for_backward({a, b});
        return gemm_forward(a, b, ndim, batch_ndim, trans_a, trans_b);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        auto const a = saved[0];
        auto const b = saved[1];
        int64_t const ndim = ctx->saved_data["ndim"].toInt();
        int64_t const batch_ndim =
            ctx->saved_data["batch_ndim"].toInt();
        bool const trans_a = ctx->saved_data["trans_a"].toBool();
        bool const trans_b = ctx->saved_data["trans_b"].toBool();
        std::array<bool, 2> const mask = {
            ctx->needs_input_grad(0),
            ctx->needs_input_grad(1),
        };
        auto gb = gemm_backward(
            a,
            b,
            grad_outputs[0],
            ndim,
            batch_ndim,
            mask,
            trans_a,
            trans_b);
        torch::autograd::variable_list result(6);
        result[0] = std::get<0>(gb);
        result[1] = std::get<1>(gb);
        return result;
    }
};

} // namespace detail

//! Differentiable N-D GEMM (Python ``torch_nntile.gemm``).
inline at::Tensor gemm(
    const at::Tensor &a,
    const at::Tensor &b,
    int64_t ndim,
    int64_t batch_ndim = 0,
    bool trans_a = false,
    bool trans_b = false)
{
    return detail::GemmFn::apply(
        a,
        b,
        ndim,
        batch_ndim,
        trans_a,
        trans_b);
}

} // namespace torch_nntile
