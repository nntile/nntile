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
    TORCH_CHECK(tensor.is_contiguous(), "nntile layer_norm requires contiguous");
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
    pin_graph_op_inputs(inputs);
    pin_graph_op_output(output, false);
    pin_graph_op_output(mean, false);
    pin_graph_op_output(rstd, false);
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
    check_norm_tensor(input, "input");
    const int64_t norm_axis = resolve_norm_axis(input.sizes(), normalized_shape);
    check_optional_affine(input, weight, bias, norm_axis);

    at::Tensor output = at::empty_like(input);
    // Reduced (non-keepdim) stats — matches C++ ``NNLayerNormOp`` buffers and
    // avoids ``scale_slice`` broadcast of mean/rstd.
    const auto stats_sizes = reduced_sizes(input.sizes(), norm_axis);
    at::Tensor mean = at::empty(
        stats_sizes,
        input.options().memory_format(at::MemoryFormat::Contiguous));
    at::Tensor rstd = at::empty(
        stats_sizes,
        input.options().memory_format(at::MemoryFormat::Contiguous));
    run_layer_norm_forward(
        input, weight, bias, norm_axis, eps, output, mean, rstd);
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
    check_norm_tensor(grad_out, "grad_out");
    check_norm_tensor(input, "input");
    check_norm_tensor(mean, "mean");
    check_norm_tensor(rstd, "rstd");
    const int64_t norm_axis = resolve_norm_axis(input.sizes(), normalized_shape);
    check_optional_affine(input, weight, bias, norm_axis);

    // Forward now saves reduced stats; accept legacy keepdim tensors too.
    at::Tensor mean_reduced = mean;
    at::Tensor rstd_reduced = rstd;
    if (mean.dim() == input.dim() && mean.size(norm_axis) == 1)
    {
        mean_reduced = mean.squeeze(norm_axis);
    }
    if (rstd.dim() == input.dim() && rstd.size(norm_axis) == 1)
    {
        rstd_reduced = rstd.squeeze(norm_axis);
    }

    at::Tensor grad_input;
    at::Tensor grad_weight;
    at::Tensor grad_bias;
    std::vector<at::Tensor> inputs = {grad_out, input, mean_reduced, rstd_reduced};
    if (weight.has_value())
    {
        inputs.push_back(*weight);
    }
    pin_graph_op_inputs(inputs);

    if (output_mask[0])
    {
        grad_input = at::empty_like(input);
        pin_graph_op_output(grad_input, false);
    }
    if (output_mask[1] && weight.has_value())
    {
        grad_weight = at::empty_like(*weight);
        pin_graph_op_output(grad_weight, false);
    }
    if (output_mask[2] && bias.has_value())
    {
        grad_bias = at::empty_like(*bias);
        pin_graph_op_output(grad_bias, false);
    }

    tensor_layer_norm_backward_fp32(
        grad_out,
        input,
        mean_reduced,
        rstd_reduced,
        weight.has_value() ? &*weight : nullptr,
        weight.has_value(),
        bias.has_value(),
        output_mask[0] ? &grad_input : nullptr,
        output_mask[1] && weight.has_value() ? &grad_weight : nullptr,
        output_mask[2] && bias.has_value() ? &grad_bias : nullptr,
        output_mask[0],
        output_mask[1] && weight.has_value(),
        output_mask[2] && bias.has_value(),
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
