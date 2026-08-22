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
}

// StarPU kernels need dense NCHW; ``cat`` / view grads may be strided.
at::Tensor as_contiguous_fp32(const at::Tensor &tensor, const char *name)
{
    check_conv_tensor(tensor, name);
    return tensor.is_contiguous() ? tensor : tensor.contiguous();
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

std::vector<int64_t> expand_spatial_2d(
    c10::IntArrayRef values,
    int64_t default0,
    int64_t default1)
{
    if (values.empty())
    {
        return {default0, default1};
    }
    if (values.size() == 1)
    {
        return {values[0], values[0]};
    }
    TORCH_CHECK(
        values.size() == 2,
        "nntile convolution: expected 1D or 2D spatial args");
    return {values[0], values[1]};
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
    const std::vector<int64_t> stride_2d =
        expand_spatial_2d(stride, 1, 1);
    const std::vector<int64_t> padding_2d =
        expand_spatial_2d(padding, 0, 0);
    const std::vector<int64_t> dilation_2d =
        expand_spatial_2d(dilation, 1, 1);
    const std::vector<int64_t> output_padding_2d =
        expand_spatial_2d(output_padding, 0, 0);

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
                stride_2d[dim],
                padding_2d[dim],
                dilation_2d[dim],
                output_padding_2d[dim]);
        }
        else
        {
            out_shape[idx] = conv_output_extent(
                input.size(idx),
                weight.size(idx),
                stride_2d[dim],
                padding_2d[dim],
                dilation_2d[dim]);
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
    nntile::GraphFillScope record;
    const at::Tensor input_c =
        as_contiguous_fp32(input, "nntile convolution input");
    const at::Tensor weight_c =
        as_contiguous_fp32(weight, "nntile convolution weight");
    // TORCH_FN may wrap Python ``None`` as an undefined Tensor inside
    // optional rather than ``nullopt``.
    const bool has_bias = bias.has_value() && bias->defined();
    at::Tensor bias_c;
    if (has_bias)
    {
        bias_c = as_contiguous_fp32(*bias, "nntile convolution bias");
    }
    std::vector<int64_t> stride_i =
        expand_spatial_2d(sym_to_i64(stride), 1, 1);
    std::vector<int64_t> padding_i =
        expand_spatial_2d(sym_to_i64(padding), 0, 0);
    std::vector<int64_t> dilation_i =
        expand_spatial_2d(sym_to_i64(dilation), 1, 1);
    std::vector<int64_t> output_padding_i =
        expand_spatial_2d(sym_to_i64(output_padding), 0, 0);
    std::vector<int64_t> out_shape = convolution_output_shape(
        input_c,
        weight_c,
        stride_i,
        padding_i,
        dilation_i,
        transposed,
        output_padding_i,
        groups.expect_int());
    at::Tensor out = empty_metadata_tensor(
        out_shape,
        input_c.scalar_type(),
        input_c.device());
    tensor_convolution_fp32(
        input_c,
        weight_c,
        has_bias ? &bias_c : nullptr,
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
    nntile::GraphFillScope record;
    const at::Tensor grad_c = as_contiguous_fp32(
        grad_output,
        "nntile convolution_backward grad");
    const at::Tensor input_c = as_contiguous_fp32(
        input,
        "nntile convolution_backward input");
    const at::Tensor weight_c = as_contiguous_fp32(
        weight,
        "nntile convolution_backward weight");
    std::vector<int64_t> stride_i =
        expand_spatial_2d(sym_to_i64(stride), 1, 1);
    std::vector<int64_t> padding_i =
        expand_spatial_2d(sym_to_i64(padding), 0, 0);
    std::vector<int64_t> dilation_i =
        expand_spatial_2d(sym_to_i64(dilation), 1, 1);
    std::vector<int64_t> output_padding_i =
        expand_spatial_2d(sym_to_i64(output_padding), 0, 0);
    at::Tensor grad_input;
    at::Tensor grad_weight;
    at::Tensor grad_bias;
    if (output_mask[0])
    {
        grad_input = empty_metadata_tensor(
            input_c.sizes(),
            input_c.scalar_type(),
            input_c.device());
    }
    if (output_mask[1])
    {
        grad_weight = empty_metadata_tensor(
            weight_c.sizes(),
            weight_c.scalar_type(),
            weight_c.device());
    }
    if (output_mask[2])
    {
        grad_bias = empty_metadata_tensor(
            {grad_c.size(1)},
            grad_c.scalar_type(),
            grad_c.device());
    }
    tensor_convolution_backward_fp32(
        grad_c,
        input_c,
        weight_c,
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
