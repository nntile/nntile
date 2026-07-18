/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_max_pool2d.cpp
 */

#include "nntile_executor.h"
#include "nntile_tensor_gc.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <torch/library.h>

#include <vector>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

void check_fp32(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(is_nntile_device(tensor.device()), name, ": expected nntile");
    TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float, name, ": fp32");
    TORCH_CHECK(tensor.is_contiguous(), name, ": contiguous tensor required");
}

void check_indices(const at::Tensor &tensor)
{
    TORCH_CHECK(is_nntile_device(tensor.device()), "max_pool indices nntile");
    TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Long, "i64 indices");
    TORCH_CHECK(tensor.is_contiguous(), "max_pool indices contiguous");
}

std::vector<int64_t> meta_max_pool_shape(
    const at::Tensor &input,
    at::IntArrayRef kernel,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode)
{
    at::Tensor meta = at::empty(
        input.sizes(),
        input.options().device(c10::DeviceType::Meta));
    auto result = at::max_pool2d_with_indices(
        meta,
        kernel,
        stride,
        padding,
        dilation,
        ceil_mode);
    return std::get<0>(result).sizes().vec();
}

} // namespace

std::tuple<at::Tensor, at::Tensor> max_pool2d_with_indices(
    const at::Tensor &input,
    at::IntArrayRef kernel,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode)
{
    check_fp32(input, "nntile max_pool2d_with_indices");
    std::vector<int64_t> out_shape = meta_max_pool_shape(
        input,
        kernel,
        stride,
        padding,
        dilation,
        ceil_mode);
    at::Tensor out = empty_metadata_tensor(
        out_shape,
        input.scalar_type(),
        input.device());
    at::Tensor indices = empty_metadata_tensor(
        out_shape,
        at::ScalarType::Long,
        input.device());
    tensor_max_pool2d_with_indices_fp32(
        input,
        out,
        indices,
        kernel,
        stride,
        padding,
        dilation,
        ceil_mode);
    return {out, indices};
}

std::tuple<at::Tensor &, at::Tensor &> max_pool2d_with_indices_out(
    const at::Tensor &input,
    at::IntArrayRef kernel,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    at::Tensor &out,
    at::Tensor &indices)
{
    check_fp32(input, "nntile max_pool2d_with_indices.out");
    check_fp32(out, "nntile max_pool2d_with_indices.out output");
    check_indices(indices);
    tensor_max_pool2d_with_indices_fp32(
        input,
        out,
        indices,
        kernel,
        stride,
        padding,
        dilation,
        ceil_mode);
    return {out, indices};
}

at::Tensor max_pool2d_with_indices_backward(
    const at::Tensor &grad_output,
    const at::Tensor &input,
    at::IntArrayRef kernel,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor &indices)
{
    check_fp32(grad_output, "nntile max_pool2d_backward grad");
    check_fp32(input, "nntile max_pool2d_backward input");
    check_indices(indices);
    at::Tensor grad_input = empty_metadata_tensor(
        input.sizes(),
        input.scalar_type(),
        input.device());
    tensor_max_pool2d_with_indices_backward_fp32(
        grad_output,
        input,
        indices,
        grad_input,
        kernel,
        stride,
        padding,
        dilation,
        ceil_mode);
    return grad_input;
}

at::Tensor &max_pool2d_with_indices_backward_out(
    const at::Tensor &grad_output,
    const at::Tensor &input,
    at::IntArrayRef kernel,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor &indices,
    at::Tensor &grad_input)
{
    check_fp32(grad_output, "nntile max_pool2d_backward.out grad");
    check_fp32(input, "nntile max_pool2d_backward.out input");
    check_indices(indices);
    tensor_max_pool2d_with_indices_backward_fp32(
        grad_output,
        input,
        indices,
        grad_input,
        kernel,
        stride,
        padding,
        dilation,
        ceil_mode);
    return grad_input;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(
        "max_pool2d_with_indices",
        TORCH_FN(torch_nntile::max_pool2d_with_indices));
    m.impl(
        "max_pool2d_with_indices.out",
        TORCH_FN(torch_nntile::max_pool2d_with_indices_out));
    m.impl(
        "max_pool2d_with_indices_backward",
        TORCH_FN(torch_nntile::max_pool2d_with_indices_backward));
    m.impl(
        "max_pool2d_with_indices_backward.grad_input",
        TORCH_FN(torch_nntile::max_pool2d_with_indices_backward_out));
}
