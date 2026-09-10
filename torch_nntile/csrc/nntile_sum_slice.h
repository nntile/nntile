/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_sum_slice.h
 * Differentiable ``sum_slice`` (old wrappers ``nntile.layer.gap.GAP``).
 */

#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/autograd/custom_function.h>

#include <cstdint>
#include <vector>

namespace torch_nntile
{

at::Tensor sum_slice_forward(
    const at::Tensor &src,
    int64_t axis,
    double alpha = 1.0,
    double beta = 0.0);

at::Tensor sum_slice_backward(
    const at::Tensor &grad_out,
    at::IntArrayRef src_sizes,
    int64_t axis,
    double alpha = 1.0);

namespace detail
{

class SumSliceFn : public torch::autograd::Function<SumSliceFn>
{
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        at::Tensor src,
        int64_t axis,
        double alpha,
        double beta)
    {
        ctx->saved_data["axis"] = axis;
        ctx->saved_data["alpha"] = alpha;
        ctx->saved_data["src_sizes"] = at::tensor(
            src.sizes().vec(),
            at::TensorOptions().dtype(at::kLong).device(at::kCPU));
        return sum_slice_forward(src, axis, alpha, beta);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        int64_t const axis = ctx->saved_data["axis"].toInt();
        double const alpha = ctx->saved_data["alpha"].toDouble();
        at::Tensor sizes_t =
            ctx->saved_data["src_sizes"].toTensor().contiguous();
        std::vector<int64_t> src_sizes(
            sizes_t.data_ptr<int64_t>(),
            sizes_t.data_ptr<int64_t>() + sizes_t.numel());
        torch::autograd::variable_list result(4);
        if (ctx->needs_input_grad(0))
        {
            result[0] = sum_slice_backward(
                grad_outputs[0],
                src_sizes,
                axis,
                alpha);
        }
        return result;
    }
};

} // namespace detail

//! Differentiable ``sum_slice`` (Python ``torch_nntile.sum_slice``).
inline at::Tensor sum_slice(
    const at::Tensor &src,
    int64_t axis,
    double alpha = 1.0,
    double beta = 0.0)
{
    return detail::SumSliceFn::apply(src, axis, alpha, beta);
}

//! Global average pool over axis 0 (old ``GAP`` without the side-R transpose).
inline at::Tensor gap(const at::Tensor &x)
{
    double const alpha = 1.0 / static_cast<double>(x.size(0));
    return sum_slice(x, /*axis=*/0, alpha, /*beta=*/0.0);
}

} // namespace torch_nntile
