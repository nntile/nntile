/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_rms_norm.cpp
 */

#include "nntile_rms_norm.h"

#include "nntile_executor.h"
#include "nntile_graph_recorder_impl.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>

#include <array>
#include <cmath>
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

int64_t resolve_norm_axis(
    c10::IntArrayRef input_shape,
    c10::IntArrayRef normalized_shape)
{
    TORCH_CHECK(
        normalized_shape.size() == 1,
        "nntile rms_norm supports a single normalized dimension");
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

float resolve_eps(std::optional<double> eps)
{
    if (eps.has_value())
    {
        return static_cast<float>(*eps);
    }
    return std::numeric_limits<float>::epsilon();
}

} // namespace

std::tuple<at::Tensor, at::Tensor> rms_norm_forward(
    const at::Tensor &input,
    at::IntArrayRef normalized_shape,
    const std::optional<at::Tensor> &weight,
    std::optional<double> eps)
{
    check_norm_tensor(input, "input");
    const int64_t norm_axis = resolve_norm_axis(input.sizes(), normalized_shape);
    if (weight.has_value())
    {
        check_norm_tensor(*weight, "weight");
        TORCH_CHECK(
            weight->dim() == 1 &&
                weight->size(0) == input.size(norm_axis),
            "nntile rms_norm: invalid weight shape");
    }

    at::Tensor output = at::empty_like(input);
    at::Tensor rstd = at::empty(
        reduced_sizes(input.sizes(), norm_axis),
        input.options().memory_format(at::MemoryFormat::Contiguous));

    std::vector<at::Tensor> inputs = {input};
    if (weight.has_value())
    {
        inputs.push_back(*weight);
    }
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

std::tuple<at::Tensor, at::Tensor> rms_norm_backward(
    const at::Tensor &grad_out,
    const at::Tensor &input,
    at::IntArrayRef normalized_shape,
    const at::Tensor &rstd,
    const std::optional<at::Tensor> &weight,
    std::array<bool, 2> output_mask)
{
    check_norm_tensor(grad_out, "grad_out");
    check_norm_tensor(input, "input");
    check_norm_tensor(rstd, "rstd");
    const int64_t norm_axis = resolve_norm_axis(input.sizes(), normalized_shape);
    if (weight.has_value())
    {
        check_norm_tensor(*weight, "weight");
    }

    at::Tensor rstd_reduced = rstd;
    if (rstd.dim() == input.dim() && rstd.size(norm_axis) == 1)
    {
        rstd_reduced = rstd.squeeze(norm_axis);
    }

    at::Tensor grad_input;
    at::Tensor grad_weight;
    std::vector<at::Tensor> inputs = {grad_out, input, rstd_reduced};
    if (weight.has_value())
    {
        inputs.push_back(*weight);
    }

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
        rstd_reduced,
        weight.has_value() ? &*weight : nullptr,
        weight.has_value(),
        output_mask[0] ? &grad_input : nullptr,
        output_mask[1] && weight.has_value() ? &grad_weight : nullptr,
        output_mask[0],
        output_mask[1] && weight.has_value(),
        norm_axis);

    return {grad_input, grad_weight};
}

} // namespace torch_nntile
