/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_avg_pool2d.cpp
 */

#include "nntile_executor.h"
#include "nntile_tensor_gc.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <torch/library.h>

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

void check_pool_input(const at::Tensor &input, const char *name)
{
    TORCH_CHECK(is_nntile_device(input.device()), name, ": expected nntile");
    TORCH_CHECK(
        input.scalar_type() == at::ScalarType::Float,
        name,
        ": float32 only");
    TORCH_CHECK(input.is_contiguous(), name, ": contiguous input required");
}

std::vector<int64_t> meta_avg_pool_shape(
    const at::Tensor &input,
    at::IntArrayRef kernel,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override)
{
    at::Tensor meta = at::empty(
        input.sizes(),
        input.options().device(c10::DeviceType::Meta));
    at::Tensor out = at::avg_pool2d(
        meta,
        kernel,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
    return out.sizes().vec();
}

} // namespace

at::Tensor avg_pool2d(
    const at::Tensor &input,
    at::IntArrayRef kernel,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override)
{
    check_pool_input(input, "nntile avg_pool2d");
    std::vector<int64_t> out_shape = meta_avg_pool_shape(
        input,
        kernel,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
    at::Tensor out = empty_metadata_tensor(
        out_shape,
        input.scalar_type(),
        input.device());
    tensor_avg_pool2d_fp32(
        input,
        out,
        kernel,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
    return out;
}

at::Tensor &avg_pool2d_out(
    const at::Tensor &input,
    at::IntArrayRef kernel,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override,
    at::Tensor &out)
{
    check_pool_input(input, "nntile avg_pool2d.out");
    TORCH_CHECK(is_nntile_device(out.device()), "avg_pool2d.out: nntile out");
    TORCH_CHECK(out.scalar_type() == at::ScalarType::Float, "fp32 out only");
    tensor_avg_pool2d_fp32(
        input,
        out,
        kernel,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
    return out;
}

at::Tensor avg_pool2d_backward(
    const at::Tensor &grad_output,
    const at::Tensor &input,
    at::IntArrayRef kernel,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override)
{
    check_pool_input(input, "nntile avg_pool2d_backward");
    check_pool_input(grad_output, "nntile avg_pool2d_backward grad");
    at::Tensor grad_input = empty_metadata_tensor(
        input.sizes(),
        input.scalar_type(),
        input.device());
    tensor_avg_pool2d_backward_fp32(
        grad_output,
        input,
        grad_input,
        kernel,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
    return grad_input;
}

at::Tensor &avg_pool2d_backward_out(
    const at::Tensor &grad_output,
    const at::Tensor &input,
    at::IntArrayRef kernel,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override,
    at::Tensor &grad_input)
{
    check_pool_input(input, "nntile avg_pool2d_backward.out input");
    check_pool_input(grad_output, "nntile avg_pool2d_backward.out grad");
    TORCH_CHECK(
        is_nntile_device(grad_input.device()),
        "avg_pool2d_backward.out: nntile grad_input");
    tensor_avg_pool2d_backward_fp32(
        grad_output,
        input,
        grad_input,
        kernel,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
    return grad_input;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("avg_pool2d", TORCH_FN(torch_nntile::avg_pool2d));
    m.impl("avg_pool2d.out", TORCH_FN(torch_nntile::avg_pool2d_out));
    m.impl(
        "avg_pool2d_backward",
        TORCH_FN(torch_nntile::avg_pool2d_backward));
    m.impl(
        "avg_pool2d_backward.grad_input",
        TORCH_FN(torch_nntile::avg_pool2d_backward_out));
}
