/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_sdpa.h
 */

#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/autograd/custom_function.h>

#include <cstdint>
#include <optional>
#include <tuple>

namespace torch_nntile
{

at::Tensor sdpa_forward(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const std::optional<at::Tensor> &mask,
    int64_t batch_ndim,
    bool is_causal = false);

std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_backward(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &grad_out,
    const std::optional<at::Tensor> &mask,
    int64_t batch_ndim,
    bool is_causal = false);

namespace detail
{

class SdpaKernelFn : public torch::autograd::Function<SdpaKernelFn>
{
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        at::Tensor q,
        at::Tensor k,
        at::Tensor v,
        at::Tensor mask,
        int64_t batch_ndim,
        bool has_mask)
    {
        ctx->saved_data["batch_ndim"] = batch_ndim;
        ctx->saved_data["has_mask"] = has_mask;
        std::optional<at::Tensor> mask_opt;
        if (has_mask)
        {
            mask_opt = mask;
            ctx->save_for_backward({q, k, v, mask});
        }
        else
        {
            ctx->save_for_backward({q, k, v});
        }
        return sdpa_forward(q, k, v, mask_opt, batch_ndim);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        int64_t const batch_ndim =
            ctx->saved_data["batch_ndim"].toInt();
        bool const has_mask =
            ctx->saved_data["has_mask"].toBool();
        std::optional<at::Tensor> mask_opt;
        if (has_mask)
        {
            mask_opt = saved[3];
        }
        auto gb = sdpa_backward(
            saved[0],
            saved[1],
            saved[2],
            grad_outputs[0],
            mask_opt,
            batch_ndim);
        torch::autograd::variable_list result(6);
        result[0] = std::get<0>(gb);
        result[1] = std::get<1>(gb);
        result[2] = std::get<2>(gb);
        return result;
    }
};

} // namespace detail

//! SDPA on kernel layout ``[n_heads, batch, seq, head_size]``.
inline at::Tensor sdpa_kernel(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const std::optional<at::Tensor> &mask = std::nullopt,
    int64_t batch_ndim = 2)
{
    if (mask.has_value())
    {
        return detail::SdpaKernelFn::apply(
            q,
            k,
            v,
            *mask,
            batch_ndim,
            true);
    }
    // No mask: still use libnntile via SdpaKernelFn (not ATen SDPA).
    at::Tensor dummy_mask = at::empty(
        {0},
        at::TensorOptions().dtype(at::kBool).device(q.device()));
    return detail::SdpaKernelFn::apply(
        q,
        k,
        v,
        dummy_mask,
        batch_ndim,
        false);
}

} // namespace torch_nntile
