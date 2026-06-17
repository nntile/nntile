/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_linear.cpp
 */

#include "nntile_executor.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <torch/library.h>

#include <array>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

void check_linear_tensors(
    const at::Tensor &input,
    const at::Tensor &weight,
    const std::optional<at::Tensor> &bias,
    const std::optional<at::Tensor> &out = std::nullopt)
{
    TORCH_CHECK(
        is_nntile_device(input.device()) &&
            is_nntile_device(weight.device()),
        "nntile linear expects input and weight on device nntile");
    if (out.has_value())
    {
        TORCH_CHECK(
            is_nntile_device(out->device()),
            "nntile linear.out expects output on device nntile");
    }
    TORCH_CHECK(input.dim() >= 1, "nntile linear: input must be at least 1D");
    TORCH_CHECK(weight.dim() == 2, "nntile linear: weight must be 2D");
    TORCH_CHECK(
        input.size(-1) == weight.size(1),
        "nntile linear: feature dimension mismatch");
    TORCH_CHECK(
        input.scalar_type() == at::ScalarType::Float &&
            weight.scalar_type() == at::ScalarType::Float,
        "nntile linear supports float32 only");
    TORCH_CHECK(
        input.is_contiguous() && weight.is_contiguous(),
        "nntile linear requires contiguous tensors");
    if (bias.has_value())
    {
        TORCH_CHECK(false, "nntile linear: bias is not supported");
    }
}

at::Tensor make_linear_output(
    const at::Tensor &input,
    const at::Tensor &weight)
{
    auto out_sizes = input.sizes().vec();
    out_sizes.back() = weight.size(0);
    return at::empty(
        out_sizes,
        input.options().memory_format(at::MemoryFormat::Contiguous));
}

void run_linear(
    const at::Tensor &input,
    const at::Tensor &weight,
    at::Tensor &output)
{
    tensor_linear_fp32(
        input.data_ptr<float>(),
        input.sizes(),
        weight.data_ptr<float>(),
        weight.sizes(),
        output.data_ptr<float>(),
        output.sizes());
}

} // namespace

at::Tensor linear(
    const at::Tensor &input,
    const at::Tensor &weight,
    const std::optional<at::Tensor> &bias)
{
    check_linear_tensors(input, weight, bias);
    at::Tensor output = make_linear_output(input, weight);
    run_linear(input, weight, output);
    return output;
}

at::Tensor &linear_out(
    const at::Tensor &input,
    const at::Tensor &weight,
    const std::optional<at::Tensor> &bias,
    at::Tensor &out)
{
    check_linear_tensors(input, weight, bias, out);
    TORCH_CHECK(
        out.sizes() == make_linear_output(input, weight).sizes(),
        "nntile linear.out: output shape mismatch");
    TORCH_CHECK(out.is_contiguous(), "nntile linear.out requires contiguous out");
    run_linear(input, weight, out);
    return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_backward(
    const at::Tensor &input,
    const at::Tensor &grad_output,
    const at::Tensor &weight,
    std::array<bool, 3> output_mask)
{
    TORCH_CHECK(
        is_nntile_device(input.device()) &&
            is_nntile_device(grad_output.device()) &&
            is_nntile_device(weight.device()),
        "nntile linear_backward expects nntile tensors");
    TORCH_CHECK(
        input.scalar_type() == at::ScalarType::Float &&
            grad_output.scalar_type() == at::ScalarType::Float &&
            weight.scalar_type() == at::ScalarType::Float,
        "nntile linear_backward supports float32 only");
    TORCH_CHECK(
        input.is_contiguous() && grad_output.is_contiguous() &&
            weight.is_contiguous(),
        "nntile linear_backward requires contiguous tensors");
    TORCH_CHECK(!output_mask[2], "nntile linear_backward: bias is not supported");

    at::Tensor grad_input;
    at::Tensor grad_weight;
    if (output_mask[0])
    {
        grad_input = at::empty_like(input);
        tensor_linear_backward_input_fp32(
            grad_output.data_ptr<float>(),
            grad_output.sizes(),
            weight.data_ptr<float>(),
            weight.sizes(),
            grad_input.data_ptr<float>(),
            grad_input.sizes());
    }
    if (output_mask[1])
    {
        grad_weight = at::empty_like(weight);
        tensor_linear_backward_weight_fp32(
            grad_output.data_ptr<float>(),
            grad_output.sizes(),
            input.data_ptr<float>(),
            input.sizes(),
            grad_weight.data_ptr<float>(),
            grad_weight.sizes());
    }
    return {grad_input, grad_weight, at::Tensor()};
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("linear", TORCH_FN(torch_nntile::linear));
    m.impl("linear.out", TORCH_FN(torch_nntile::linear_out));
    m.impl("linear_backward", TORCH_FN(torch_nntile::linear_backward));
}
