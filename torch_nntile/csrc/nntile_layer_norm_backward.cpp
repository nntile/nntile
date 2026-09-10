/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_layer_norm_backward.cpp
 * PrivateUse1 ``aten::native_layer_norm_backward``.
 *
 * CompositeExplicit ``native_layer_norm`` is PyTorch's
 * ``math_native_layer_norm`` (reshape + ``native_batch_norm`` +
 * affine). Core has no composite for the backward schema (CPU/CUDA
 * kernels only), so VariableType would miss a PrivateUse1 kernel.
 * This is that math dual: affine grads via ``add`` / ``mul`` / ``sum``,
 * input grad via ``native_batch_norm_backward``. Not a fused StarPU
 * LayerNorm codelet.
 */

#include "nntile_no_implicit_copy.h"

#include <nntile/tensor/graph_fill_timer.hh>

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <torch/library.h>

#include <array>
#include <vector>

namespace torch_nntile
{

namespace
{

at::Tensor optional_defined(const std::optional<at::Tensor> &tensor)
{
    if (tensor.has_value() && tensor->defined())
    {
        return *tensor;
    }
    return at::Tensor();
}

int64_t outer_count(const at::Tensor &input, int64_t axis)
{
    int64_t m = 1;
    for (int64_t i = 0; i < axis; ++i)
    {
        m *= input.size(i);
    }
    return m;
}

std::vector<int64_t> stat_keepdim_shape(
    const at::Tensor &input,
    int64_t axis)
{
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(input.dim()));
    for (int64_t i = 0; i < axis; ++i)
    {
        shape.push_back(input.size(i));
    }
    for (int64_t i = axis; i < input.dim(); ++i)
    {
        shape.push_back(1);
    }
    return shape;
}

at::Tensor reshape_stat(
    const at::Tensor &stat,
    const at::Tensor &input,
    int64_t axis)
{
    const int64_t m = outer_count(input, axis);
    TORCH_CHECK(
        stat.numel() == m,
        "nntile native_layer_norm_backward: stat numel");
    return stat.reshape(stat_keepdim_shape(input, axis));
}

at::Tensor sum_leading(const at::Tensor &src, int64_t axis)
{
    if (axis == 0)
    {
        return src;
    }
    std::vector<int64_t> dims;
    dims.reserve(static_cast<size_t>(axis));
    for (int64_t i = 0; i < axis; ++i)
    {
        dims.push_back(i);
    }
    return at::sum(src, dims, /*keepdim=*/false);
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor>
native_layer_norm_backward(
    const at::Tensor &grad_out,
    const at::Tensor &input,
    at::IntArrayRef normalized_shape,
    const at::Tensor &mean,
    const at::Tensor &rstd,
    const std::optional<at::Tensor> &weight_opt,
    const std::optional<at::Tensor> &bias_opt,
    std::array<bool, 3> output_mask)
{
    nntile::GraphFillScope record;
    require_nntile_operand(
        grad_out,
        "native_layer_norm_backward",
        "grad");
    require_nntile_operand(
        input,
        "native_layer_norm_backward",
        "input");
    require_nntile_operand(
        mean,
        "native_layer_norm_backward",
        "mean");
    require_nntile_operand(
        rstd,
        "native_layer_norm_backward",
        "rstd");
    TORCH_CHECK(
        grad_out.scalar_type() == at::kFloat
            && input.scalar_type() == at::kFloat,
        "nntile native_layer_norm_backward: float32 only");
    TORCH_CHECK(
        static_cast<int64_t>(normalized_shape.size()) <= input.dim(),
        "nntile native_layer_norm_backward: normalized_shape rank");
    const int64_t axis =
        input.dim() - static_cast<int64_t>(normalized_shape.size());
    for (size_t i = 0; i < normalized_shape.size(); ++i)
    {
        TORCH_CHECK(
            input.size(axis + static_cast<int64_t>(i))
                == normalized_shape[i],
            "nntile native_layer_norm_backward: shape mismatch");
    }

    const at::Tensor weight = optional_defined(weight_opt);
    const at::Tensor bias = optional_defined(bias_opt);
    if (weight.defined())
    {
        require_nntile_operand(
            weight,
            "native_layer_norm_backward",
            "weight");
    }
    if (bias.defined())
    {
        require_nntile_operand(
            bias,
            "native_layer_norm_backward",
            "bias");
    }

    const int64_t m = outer_count(input, axis);
    at::Tensor d_x_hat = grad_out;
    at::Tensor d_weight;
    at::Tensor d_bias;
    if (weight.defined())
    {
        d_x_hat = at::mul(grad_out, weight);
        if (output_mask[1])
        {
            at::Tensor mean_k = reshape_stat(mean, input, axis);
            at::Tensor rstd_k = reshape_stat(rstd, input, axis);
            at::Tensor x_hat = at::mul(
                at::add(input, mean_k, /*alpha=*/-1),
                rstd_k);
            d_weight = sum_leading(at::mul(grad_out, x_hat), axis);
        }
    }
    if (bias.defined() && output_mask[2])
    {
        d_bias = sum_leading(grad_out, axis);
    }

    at::Tensor d_input;
    if (output_mask[0] && input.numel() > 0 && m > 0)
    {
        at::Tensor input_bn = input.reshape({1, m, -1});
        at::Tensor dx_bn = d_x_hat.reshape({1, m, -1});
        at::Tensor mean_flat = mean.reshape({m});
        at::Tensor rstd_flat = rstd.reshape({m});
        auto bn_grads = at::native_batch_norm_backward(
            dx_bn,
            input_bn,
            /*weight=*/std::nullopt,
            /*running_mean=*/std::nullopt,
            /*running_var=*/std::nullopt,
            mean_flat,
            rstd_flat,
            /*training=*/true,
            /*eps=*/1e-5,
            std::array<bool, 3>{true, false, false});
        d_input = std::get<0>(bn_grads).reshape(input.sizes());
    }
    else if (output_mask[0])
    {
        d_input = at::empty_like(input);
    }
    return {d_input, d_weight, d_bias};
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(
        "native_layer_norm_backward",
        TORCH_FN(torch_nntile::native_layer_norm_backward));
}
