/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_rms_norm_aten.cpp
 */

#include "nntile_executor.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <c10/core/SymIntArrayRef.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <array>
#include <limits>
#include <optional>
#include <vector>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

void check_norm_tensor(
    const at::Tensor &tensor,
    const char *name)
{
    TORCH_CHECK(
        is_nntile_device(tensor.device()),
        "nntile rms_norm: expected nntile ",
        name);
    TORCH_CHECK(
        tensor.scalar_type() == at::ScalarType::Float,
        "nntile rms_norm supports float32 only");
    TORCH_CHECK(tensor.is_contiguous(), "nntile rms_norm requires contiguous");
}

std::vector<int64_t> to_int_vec(at::IntArrayRef shape)
{
    return std::vector<int64_t>(shape.begin(), shape.end());
}

std::vector<int64_t> to_int_vec(c10::SymIntArrayRef shape)
{
    std::vector<int64_t> out;
    out.reserve(shape.size());
    for (const c10::SymInt &dim : shape)
    {
        out.push_back(dim.expect_int());
    }
    return out;
}

int64_t resolve_norm_axis(
    c10::IntArrayRef input_shape,
    c10::IntArrayRef normalized_shape)
{
    TORCH_CHECK(
        !normalized_shape.empty(),
        "nntile rms_norm: normalized_shape must not be empty");
    TORCH_CHECK(
        input_shape.size() >= normalized_shape.size(),
        "nntile rms_norm: input rank too small");
    const int64_t axis = static_cast<int64_t>(input_shape.size()) -
        static_cast<int64_t>(normalized_shape.size());
    for (std::size_t i = 0; i < normalized_shape.size(); ++i)
    {
        TORCH_CHECK(
            input_shape[static_cast<std::size_t>(axis) + i] ==
                normalized_shape[i],
            "nntile rms_norm: normalized_shape mismatch");
    }
    return axis;
}

std::vector<int64_t> reduced_sizes(
    c10::IntArrayRef input_shape,
    int64_t axis)
{
    std::vector<int64_t> sizes;
    sizes.reserve(static_cast<std::size_t>(axis));
    for (int64_t i = 0; i < axis; ++i)
    {
        sizes.push_back(input_shape[static_cast<std::size_t>(i)]);
    }
    return sizes;
}

float resolve_eps(std::optional<double> eps)
{
    if (eps.has_value())
    {
        return static_cast<float>(*eps);
    }
    return std::numeric_limits<float>::epsilon();
}

void check_optional_weight(
    const at::Tensor &input,
    const std::optional<at::Tensor> &weight,
    c10::IntArrayRef normalized_shape)
{
    if (!weight.has_value())
    {
        return;
    }
    check_norm_tensor(*weight, "weight");
    TORCH_CHECK(
        weight->sizes().equals(normalized_shape),
        "nntile rms_norm: invalid weight shape");
}

std::tuple<at::Tensor, at::Tensor> rms_norm_forward_impl(
    const at::Tensor &input,
    c10::IntArrayRef normalized_shape,
    const std::optional<at::Tensor> &weight,
    std::optional<double> eps)
{
    check_norm_tensor(input, "input");
    const int64_t norm_axis =
        resolve_norm_axis(input.sizes(), normalized_shape);
    check_optional_weight(input, weight, normalized_shape);

    at::Tensor output = at::empty_like(input);
    at::Tensor rstd = at::empty(
        reduced_sizes(input.sizes(), norm_axis),
        input.options().memory_format(at::MemoryFormat::Contiguous));

    tensor_rms_norm_forward_fp32(
        input,
        weight.has_value() ? &*weight : nullptr,
        weight.has_value(),
        output,
        rstd,
        norm_axis,
        resolve_eps(eps));
    return {output, rstd};
}

std::tuple<at::Tensor, at::Tensor> rms_norm_backward_impl(
    const at::Tensor &grad_out,
    const at::Tensor &input,
    c10::IntArrayRef normalized_shape,
    const at::Tensor &rstd,
    const std::optional<at::Tensor> &weight,
    std::array<bool, 2> output_mask)
{
    check_norm_tensor(grad_out, "grad_out");
    check_norm_tensor(input, "input");
    check_norm_tensor(rstd, "rstd");
    const int64_t norm_axis =
        resolve_norm_axis(input.sizes(), normalized_shape);
    check_optional_weight(input, weight, normalized_shape);

    at::Tensor grad_input;
    at::Tensor grad_weight;
    if (output_mask[0])
    {
        grad_input = at::empty_like(input);
    }
    if (output_mask[1] && weight.has_value())
    {
        grad_weight = at::empty_like(*weight);
    }

    tensor_rms_norm_backward_fp32(
        grad_out,
        input,
        rstd,
        weight.has_value() ? &*weight : nullptr,
        weight.has_value(),
        output_mask[0] ? &grad_input : nullptr,
        output_mask[1] && weight.has_value() ? &grad_weight : nullptr,
        output_mask[0],
        output_mask[1] && weight.has_value(),
        norm_axis);

    return {grad_input, grad_weight};
}

class RmsNormWeightAutograd final :
    public torch::autograd::Function<RmsNormWeightAutograd>
{
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        const at::Tensor &input,
        std::vector<int64_t> normalized_shape,
        at::Tensor weight,
        double eps)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        auto result = rms_norm_forward_impl(
            input,
            normalized_shape,
            std::optional<at::Tensor>(weight),
            eps);
        at::Tensor output = std::get<0>(result);
        at::Tensor rstd = std::get<1>(result);
        ctx->save_for_backward({input, weight, rstd});
        ctx->saved_data["normalized_shape"] = normalized_shape;
        ctx->saved_data["eps"] = eps;
        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        at::Tensor input = saved[0];
        at::Tensor weight = saved[1];
        at::Tensor rstd = saved[2];
        std::vector<int64_t> normalized_shape =
            ctx->saved_data["normalized_shape"].toIntVector();
        const bool need_gi = ctx->needs_input_grad(0);
        const bool need_gw = ctx->needs_input_grad(1);

        at::AutoDispatchBelowADInplaceOrView guard;
        auto grads = rms_norm_backward_impl(
            grad_outputs[0],
            input,
            normalized_shape,
            rstd,
            std::optional<at::Tensor>(weight),
            {need_gi, need_gw});

        at::Tensor grad_input = need_gi ? std::get<0>(grads) : at::Tensor();
        at::Tensor grad_weight = need_gw ? std::get<1>(grads) : at::Tensor();
        return {
            grad_input,
            at::Tensor(),
            grad_weight,
            at::Tensor()};
    }
};

class RmsNormNoWeightAutograd final :
    public torch::autograd::Function<RmsNormNoWeightAutograd>
{
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        const at::Tensor &input,
        std::vector<int64_t> normalized_shape,
        double eps)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        auto result = rms_norm_forward_impl(
            input,
            normalized_shape,
            std::nullopt,
            eps);
        at::Tensor output = std::get<0>(result);
        at::Tensor rstd = std::get<1>(result);
        ctx->save_for_backward({input, rstd});
        ctx->saved_data["normalized_shape"] = normalized_shape;
        ctx->saved_data["eps"] = eps;
        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        at::Tensor input = saved[0];
        at::Tensor rstd = saved[1];
        std::vector<int64_t> normalized_shape =
            ctx->saved_data["normalized_shape"].toIntVector();
        const bool need_gi = ctx->needs_input_grad(0);

        at::AutoDispatchBelowADInplaceOrView guard;
        auto grads = rms_norm_backward_impl(
            grad_outputs[0],
            input,
            normalized_shape,
            rstd,
            std::nullopt,
            {need_gi, false});

        at::Tensor grad_input = need_gi ? std::get<0>(grads) : at::Tensor();
        return {grad_input, at::Tensor(), at::Tensor()};
    }
};

} // namespace

at::Tensor rms_norm_private(
    const at::Tensor &input,
    c10::SymIntArrayRef normalized_shape,
    const std::optional<at::Tensor> &weight,
    std::optional<double> eps)
{
    const std::vector<int64_t> shape = to_int_vec(normalized_shape);
    return std::get<0>(rms_norm_forward_impl(input, shape, weight, eps));
}

at::Tensor rms_norm_autograd_private(
    const at::Tensor &input,
    c10::SymIntArrayRef normalized_shape,
    const std::optional<at::Tensor> &weight,
    std::optional<double> eps)
{
    const std::vector<int64_t> shape = to_int_vec(normalized_shape);
    const double eps_value = static_cast<double>(resolve_eps(eps));
    if (weight.has_value())
    {
        return RmsNormWeightAutograd::apply(
            input,
            shape,
            *weight,
            eps_value);
    }
    return RmsNormNoWeightAutograd::apply(input, shape, eps_value);
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("rms_norm", TORCH_FN(torch_nntile::rms_norm_private));
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m)
{
    m.impl(
        "rms_norm",
        TORCH_FN(torch_nntile::rms_norm_autograd_private));
}
