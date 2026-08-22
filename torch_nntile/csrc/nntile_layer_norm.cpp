/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_layer_norm.cpp
 */

#include "nntile_executor.h"
#include "nntile_graph_recorder_impl.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <torch/library.h>

#include <array>
#include <vector>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

at::Tensor as_contiguous_fp32(
    const at::Tensor &tensor,
    const char *name)
{
    TORCH_CHECK(
        is_nntile_device(tensor.device()),
        "nntile layer_norm: expected nntile ",
        name);
    TORCH_CHECK(
        tensor.scalar_type() == at::ScalarType::Float,
        "nntile layer_norm supports float32 only");
    return tensor.is_contiguous() ? tensor : tensor.contiguous();
}

//! Autograd may pass an undefined Tensor inside optional instead of nullopt.
std::optional<at::Tensor> optional_defined_contiguous(
    const std::optional<at::Tensor> &tensor,
    const char *name)
{
    if (!tensor.has_value() || !tensor->defined())
    {
        return std::nullopt;
    }
    return as_contiguous_fp32(*tensor, name);
}

void check_norm_tensor(
    const at::Tensor &tensor,
    const char *name)
{
    TORCH_CHECK(
        is_nntile_device(tensor.device()),
        "nntile layer_norm: expected nntile ",
        name);
    TORCH_CHECK(
        tensor.scalar_type() == at::ScalarType::Float,
        "nntile layer_norm supports float32 only");
}

int64_t resolve_norm_axis(
    c10::IntArrayRef input_shape,
    c10::IntArrayRef normalized_shape)
{
    TORCH_CHECK(
        normalized_shape.size() == 1,
        "nntile layer_norm supports a single normalized dimension");
    TORCH_CHECK(
        input_shape.size() >= normalized_shape.size(),
        "nntile layer_norm: input rank too small");
    const int64_t axis = static_cast<int64_t>(input_shape.size()) -
        static_cast<int64_t>(normalized_shape.size());
    for (std::size_t i = 0; i < normalized_shape.size(); ++i)
    {
        TORCH_CHECK(
            input_shape[static_cast<std::size_t>(axis) + i] ==
                normalized_shape[i],
            "nntile layer_norm: normalized_shape mismatch");
    }
    return axis;
}

std::vector<int64_t> reduced_sizes(
    c10::IntArrayRef input_shape,
    int64_t axis)
{
    std::vector<int64_t> sizes;
    sizes.reserve(static_cast<std::size_t>(input_shape.size()));
    for (int64_t i = 0; i < static_cast<int64_t>(input_shape.size()); ++i)
    {
        if (i != axis)
        {
            sizes.push_back(input_shape[static_cast<std::size_t>(i)]);
        }
    }
    return sizes;
}

void check_optional_affine(
    const at::Tensor &input,
    const std::optional<at::Tensor> &weight,
    const std::optional<at::Tensor> &bias,
    int64_t norm_axis)
{
    if (weight.has_value())
    {
        check_norm_tensor(*weight, "weight");
        TORCH_CHECK(
            weight->dim() == 1 &&
                weight->size(0) == input.size(norm_axis),
            "nntile layer_norm: invalid weight shape");
    }
    if (bias.has_value())
    {
        check_norm_tensor(*bias, "bias");
        TORCH_CHECK(
            bias->dim() == 1 && bias->size(0) == input.size(norm_axis),
            "nntile layer_norm: invalid bias shape");
    }
}

void run_layer_norm_forward(
    const at::Tensor &input,
    const std::optional<at::Tensor> &weight,
    const std::optional<at::Tensor> &bias,
    int64_t norm_axis,
    double eps,
    at::Tensor &output,
    at::Tensor &mean,
    at::Tensor &rstd)
{
    std::vector<at::Tensor> inputs = {input};
    if (weight.has_value())
    {
        inputs.push_back(*weight);
    }
    if (bias.has_value())
    {
        inputs.push_back(*bias);
    }
    tensor_layer_norm_forward_fp32(
        input,
        weight.has_value() ? &*weight : nullptr,
        bias.has_value() ? &*bias : nullptr,
        weight.has_value(),
        bias.has_value(),
        output,
        mean,
        rstd,
        norm_axis,
        static_cast<float>(eps));
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm(
    const at::Tensor &input,
    at::IntArrayRef normalized_shape,
    const std::optional<at::Tensor> &weight,
    const std::optional<at::Tensor> &bias,
    double eps)
{
    nntile::GraphFillScope record;
    at::Tensor input_c = as_contiguous_fp32(input, "input");
    const int64_t norm_axis = resolve_norm_axis(
        input_c.sizes(),
        normalized_shape);
    std::optional<at::Tensor> weight_c =
        optional_defined_contiguous(weight, "weight");
    std::optional<at::Tensor> bias_c =
        optional_defined_contiguous(bias, "bias");
    check_optional_affine(input_c, weight_c, bias_c, norm_axis);

    at::Tensor output = at::empty_like(input_c);
    // Reduced (non-keepdim) stats - matches C++ ``NNLayerNormOp`` buffers and
    // avoids ``scale_slice`` broadcast of mean/rstd.
    const auto stats_sizes = reduced_sizes(input_c.sizes(), norm_axis);
    at::Tensor mean = at::empty(
        stats_sizes,
        input_c.options().memory_format(at::MemoryFormat::Contiguous));
    at::Tensor rstd = at::empty(
        stats_sizes,
        input_c.options().memory_format(at::MemoryFormat::Contiguous));
    run_layer_norm_forward(
        input_c, weight_c, bias_c, norm_axis, eps, output, mean, rstd);
    return {output, mean, rstd};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm_backward(
    const at::Tensor &grad_out,
    const at::Tensor &input,
    at::IntArrayRef normalized_shape,
    const at::Tensor &mean,
    const at::Tensor &rstd,
    const std::optional<at::Tensor> &weight,
    const std::optional<at::Tensor> &bias,
    std::array<bool, 3> output_mask)
{
    nntile::GraphFillScope record;
    // Sum/Mean backward often expands a non-contiguous ones view into
    // ``grad_out``; densify before the StarPU codelet.
    at::Tensor grad_out_c = as_contiguous_fp32(grad_out, "grad_out");
    at::Tensor input_c = as_contiguous_fp32(input, "input");
    at::Tensor mean_c = as_contiguous_fp32(mean, "mean");
    at::Tensor rstd_c = as_contiguous_fp32(rstd, "rstd");
    const int64_t norm_axis = resolve_norm_axis(
        input_c.sizes(),
        normalized_shape);
    std::optional<at::Tensor> weight_c =
        optional_defined_contiguous(weight, "weight");
    std::optional<at::Tensor> bias_c =
        optional_defined_contiguous(bias, "bias");
    check_optional_affine(input_c, weight_c, bias_c, norm_axis);

    // Forward now saves reduced stats; accept legacy keepdim tensors too.
    at::Tensor mean_reduced = mean_c;
    at::Tensor rstd_reduced = rstd_c;
    if (mean_c.dim() == input_c.dim() && mean_c.size(norm_axis) == 1)
    {
        mean_reduced = mean_c.squeeze(norm_axis).contiguous();
    }
    if (rstd_c.dim() == input_c.dim() && rstd_c.size(norm_axis) == 1)
    {
        rstd_reduced = rstd_c.squeeze(norm_axis).contiguous();
    }

    at::Tensor grad_input;
    at::Tensor grad_weight;
    at::Tensor grad_bias;

    if (output_mask[0])
    {
        grad_input = at::empty_like(input_c);
    }
    if (output_mask[1] && weight_c.has_value())
    {
        grad_weight = at::empty_like(*weight_c);
    }
    if (output_mask[2] && bias_c.has_value())
    {
        grad_bias = at::empty_like(*bias_c);
    }

    tensor_layer_norm_backward_fp32(
        grad_out_c,
        input_c,
        mean_reduced,
        rstd_reduced,
        weight_c.has_value() ? &*weight_c : nullptr,
        bias_c.has_value() ? &*bias_c : nullptr,
        weight_c.has_value(),
        bias_c.has_value(),
        output_mask[0] ? &grad_input : nullptr,
        output_mask[1] && weight_c.has_value() ? &grad_weight : nullptr,
        output_mask[2] && bias_c.has_value() ? &grad_bias : nullptr,
        output_mask[0],
        output_mask[1] && weight_c.has_value(),
        output_mask[2] && bias_c.has_value(),
        norm_axis);

    return {grad_input, grad_weight, grad_bias};
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(
        "native_layer_norm",
        TORCH_FN(torch_nntile::native_layer_norm));
    m.impl(
        "native_layer_norm_backward",
        TORCH_FN(torch_nntile::native_layer_norm_backward));
}
