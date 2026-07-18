/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_convolution.cpp
 */

#include "nntile_executor.h"
#include "nntile_tensor_gc.h"

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

void check_conv_tensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(is_nntile_device(tensor.device()), name, ": expected nntile");
    TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float, name, ": fp32");
    TORCH_CHECK(tensor.is_contiguous(), name, ": contiguous tensor required");
}

std::vector<int64_t> sym_to_i64(c10::SymIntArrayRef values)
{
    std::vector<int64_t> out;
    out.reserve(values.size());
    for (const c10::SymInt &value : values)
    {
        out.push_back(value.expect_int());
    }
    return out;
}

int64_t conv_output_extent(
    int64_t input,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t dilation)
{
    return (input + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

int64_t conv_transpose_output_extent(
    int64_t input,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t output_padding)
{
    return (input - 1) * stride - 2 * padding + dilation * (kernel - 1) +
        output_padding + 1;
}

std::vector<int64_t> convolution_output_shape(
    const at::Tensor &input,
    const at::Tensor &weight,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    c10::IntArrayRef dilation,
    bool transposed,
    c10::IntArrayRef output_padding,
    int64_t groups)
{
    TORCH_CHECK(input.dim() == 4, "nntile convolution supports 4D NCHW");
    TORCH_CHECK(weight.dim() == 4, "nntile convolution weight must be 4D");
    TORCH_CHECK(stride.size() == 2, "nntile convolution stride must be 2D");
    TORCH_CHECK(padding.size() == 2, "nntile convolution padding must be 2D");
    TORCH_CHECK(
        dilation.size() == 2,
        "nntile convolution dilation must be 2D");
    TORCH_CHECK(
        output_padding.size() == 2,
        "nntile convolution output_padding must be 2D");

    std::vector<int64_t> out_shape = input.sizes().vec();
    out_shape[1] = transposed ? weight.size(1) * groups : weight.size(0);
    for (int64_t dim = 0; dim < 2; ++dim)
    {
        const int64_t idx = dim + 2;
        if (transposed)
        {
            out_shape[idx] = conv_transpose_output_extent(
                input.size(idx),
                weight.size(idx),
                stride[dim],
                padding[dim],
                dilation[dim],
                output_padding[dim]);
        }
        else
        {
            out_shape[idx] = conv_output_extent(
                input.size(idx),
                weight.size(idx),
                stride[dim],
                padding[dim],
                dilation[dim]);
        }
    }
    return out_shape;
}

} // namespace

at::Tensor convolution_overrideable(
    const at::Tensor &input,
    const at::Tensor &weight,
    const std::optional<at::Tensor> &bias,
    c10::SymIntArrayRef stride,
    c10::SymIntArrayRef padding,
    c10::SymIntArrayRef dilation,
    bool transposed,
    c10::SymIntArrayRef output_padding,
    c10::SymInt groups)
{
    check_conv_tensor(input, "nntile convolution input");
    check_conv_tensor(weight, "nntile convolution weight");
    if (bias.has_value())
    {
        check_conv_tensor(*bias, "nntile convolution bias");
    }
    std::vector<int64_t> stride_i = sym_to_i64(stride);
    std::vector<int64_t> padding_i = sym_to_i64(padding);
    std::vector<int64_t> dilation_i = sym_to_i64(dilation);
    std::vector<int64_t> output_padding_i = sym_to_i64(output_padding);
    TORCH_CHECK(stride_i.size() == 2, "nntile convolution supports 2D only");
    std::vector<int64_t> out_shape = convolution_output_shape(
        input,
        weight,
        stride_i,
        padding_i,
        dilation_i,
        transposed,
        output_padding_i,
        groups.expect_int());
    at::Tensor out = empty_metadata_tensor(
        out_shape,
        input.scalar_type(),
        input.device());
    tensor_convolution_fp32(
        input,
        weight,
        bias.has_value() ? &*bias : nullptr,
        out,
        stride_i,
        padding_i,
        dilation_i,
        transposed,
        output_padding_i,
        groups.expect_int());
    return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
convolution_backward_overrideable(
    const at::Tensor &grad_output,
    const at::Tensor &input,
    const at::Tensor &weight,
    c10::SymIntArrayRef stride,
    c10::SymIntArrayRef padding,
    c10::SymIntArrayRef dilation,
    bool transposed,
    c10::SymIntArrayRef output_padding,
    c10::SymInt groups,
    std::array<bool, 3> output_mask)
{
    check_conv_tensor(grad_output, "nntile convolution_backward grad");
    check_conv_tensor(input, "nntile convolution_backward input");
    check_conv_tensor(weight, "nntile convolution_backward weight");
    std::vector<int64_t> stride_i = sym_to_i64(stride);
    std::vector<int64_t> padding_i = sym_to_i64(padding);
    std::vector<int64_t> dilation_i = sym_to_i64(dilation);
    std::vector<int64_t> output_padding_i = sym_to_i64(output_padding);
    at::Tensor grad_input;
    at::Tensor grad_weight;
    at::Tensor grad_bias;
    if (output_mask[0])
    {
        grad_input = empty_metadata_tensor(
            input.sizes(),
            input.scalar_type(),
            input.device());
    }
    if (output_mask[1])
    {
        grad_weight = empty_metadata_tensor(
            weight.sizes(),
            weight.scalar_type(),
            weight.device());
    }
    if (output_mask[2])
    {
        grad_bias = empty_metadata_tensor(
            {grad_output.size(1)},
            grad_output.scalar_type(),
            grad_output.device());
    }
    tensor_convolution_backward_fp32(
        grad_output,
        input,
        weight,
        output_mask[0] ? &grad_input : nullptr,
        output_mask[1] ? &grad_weight : nullptr,
        output_mask[2] ? &grad_bias : nullptr,
        stride_i,
        padding_i,
        dilation_i,
        transposed,
        output_padding_i,
        groups.expect_int(),
        output_mask);
    return {grad_input, grad_weight, grad_bias};
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(
        "convolution_overrideable",
        TORCH_FN(torch_nntile::convolution_overrideable));
    m.impl(
        "convolution_backward_overrideable",
        TORCH_FN(torch_nntile::convolution_backward_overrideable));
}
