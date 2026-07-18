/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_upsample2d.cpp
 * PrivateUse1 upsample_nearest2d / upsample_bilinear2d (+ backward).
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

at::Tensor as_contiguous_fp32(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(is_nntile_device(tensor.device()), name, ": expected nntile");
    TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float, name, ": fp32");
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

std::vector<int64_t> expand_output_size_2d(c10::SymIntArrayRef output_size)
{
    std::vector<int64_t> hw = sym_to_i64(output_size);
    TORCH_CHECK(
        hw.size() == 1 || hw.size() == 2,
        "nntile upsample2d: expected 1D or 2D output_size");
    if (hw.size() == 1)
    {
        return {hw[0], hw[0]};
    }
    return hw;
}

} // namespace

at::Tensor upsample_nearest2d(
    const at::Tensor &self,
    c10::SymIntArrayRef output_size,
    std::optional<double> scales_h,
    std::optional<double> scales_w)
{
    const at::Tensor input =
        as_contiguous_fp32(self, "nntile upsample_nearest2d");
    TORCH_CHECK(input.dim() == 4, "nntile upsample_nearest2d: NCHW only");
    std::vector<int64_t> out_hw = expand_output_size_2d(output_size);
    std::vector<int64_t> out_shape = input.sizes().vec();
    out_shape[2] = out_hw[0];
    out_shape[3] = out_hw[1];
    at::Tensor out = empty_metadata_tensor(
        out_shape,
        input.scalar_type(),
        input.device());
    tensor_upsample_nearest2d_fp32(
        input,
        out,
        out_hw,
        scales_h,
        scales_w);
    return out;
}

at::Tensor &upsample_nearest2d_out(
    const at::Tensor &self,
    c10::SymIntArrayRef output_size,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    at::Tensor &out)
{
    const at::Tensor input =
        as_contiguous_fp32(self, "nntile upsample_nearest2d.out");
    TORCH_CHECK(is_nntile_device(out.device()), "upsample.out: nntile out");
    std::vector<int64_t> out_hw = expand_output_size_2d(output_size);
    tensor_upsample_nearest2d_fp32(
        input,
        out,
        out_hw,
        scales_h,
        scales_w);
    return out;
}

at::Tensor upsample_nearest2d_backward(
    const at::Tensor &grad_output,
    c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    std::optional<double> scales_h,
    std::optional<double> scales_w)
{
    const at::Tensor go = as_contiguous_fp32(
        grad_output,
        "nntile upsample_nearest2d_backward");
    std::vector<int64_t> out_hw = expand_output_size_2d(output_size);
    std::vector<int64_t> in_shape = sym_to_i64(input_size);
    TORCH_CHECK(in_shape.size() == 4, "upsample bwd: input_size NCHW");
    at::Tensor grad_input = empty_metadata_tensor(
        in_shape,
        go.scalar_type(),
        go.device());
    tensor_upsample_nearest2d_backward_fp32(
        go,
        grad_input,
        out_hw,
        in_shape,
        scales_h,
        scales_w);
    return grad_input;
}

at::Tensor &upsample_nearest2d_backward_out(
    const at::Tensor &grad_output,
    c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    at::Tensor &grad_input)
{
    const at::Tensor go = as_contiguous_fp32(
        grad_output,
        "nntile upsample_nearest2d_backward.grad_input");
    std::vector<int64_t> out_hw = expand_output_size_2d(output_size);
    std::vector<int64_t> in_shape = sym_to_i64(input_size);
    tensor_upsample_nearest2d_backward_fp32(
        go,
        grad_input,
        out_hw,
        in_shape,
        scales_h,
        scales_w);
    return grad_input;
}

at::Tensor upsample_bilinear2d(
    const at::Tensor &self,
    c10::SymIntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w)
{
    const at::Tensor input =
        as_contiguous_fp32(self, "nntile upsample_bilinear2d");
    TORCH_CHECK(input.dim() == 4, "nntile upsample_bilinear2d: NCHW only");
    std::vector<int64_t> out_hw = expand_output_size_2d(output_size);
    std::vector<int64_t> out_shape = input.sizes().vec();
    out_shape[2] = out_hw[0];
    out_shape[3] = out_hw[1];
    at::Tensor out = empty_metadata_tensor(
        out_shape,
        input.scalar_type(),
        input.device());
    tensor_upsample_bilinear2d_fp32(
        input,
        out,
        out_hw,
        align_corners,
        scales_h,
        scales_w);
    return out;
}

at::Tensor &upsample_bilinear2d_out(
    const at::Tensor &self,
    c10::SymIntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    at::Tensor &out)
{
    const at::Tensor input =
        as_contiguous_fp32(self, "nntile upsample_bilinear2d.out");
    TORCH_CHECK(is_nntile_device(out.device()), "upsample.out: nntile out");
    std::vector<int64_t> out_hw = expand_output_size_2d(output_size);
    tensor_upsample_bilinear2d_fp32(
        input,
        out,
        out_hw,
        align_corners,
        scales_h,
        scales_w);
    return out;
}

at::Tensor upsample_bilinear2d_backward(
    const at::Tensor &grad_output,
    c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w)
{
    const at::Tensor go = as_contiguous_fp32(
        grad_output,
        "nntile upsample_bilinear2d_backward");
    std::vector<int64_t> out_hw = expand_output_size_2d(output_size);
    std::vector<int64_t> in_shape = sym_to_i64(input_size);
    TORCH_CHECK(in_shape.size() == 4, "upsample bwd: input_size NCHW");
    at::Tensor grad_input = empty_metadata_tensor(
        in_shape,
        go.scalar_type(),
        go.device());
    tensor_upsample_bilinear2d_backward_fp32(
        go,
        grad_input,
        out_hw,
        in_shape,
        align_corners,
        scales_h,
        scales_w);
    return grad_input;
}

at::Tensor &upsample_bilinear2d_backward_out(
    const at::Tensor &grad_output,
    c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    at::Tensor &grad_input)
{
    const at::Tensor go = as_contiguous_fp32(
        grad_output,
        "nntile upsample_bilinear2d_backward.grad_input");
    std::vector<int64_t> out_hw = expand_output_size_2d(output_size);
    std::vector<int64_t> in_shape = sym_to_i64(input_size);
    tensor_upsample_bilinear2d_backward_fp32(
        go,
        grad_input,
        out_hw,
        in_shape,
        align_corners,
        scales_h,
        scales_w);
    return grad_input;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(
        "upsample_nearest2d",
        TORCH_FN(torch_nntile::upsample_nearest2d));
    m.impl(
        "upsample_nearest2d.out",
        TORCH_FN(torch_nntile::upsample_nearest2d_out));
    m.impl(
        "upsample_nearest2d_backward",
        TORCH_FN(torch_nntile::upsample_nearest2d_backward));
    m.impl(
        "upsample_nearest2d_backward.grad_input",
        TORCH_FN(torch_nntile::upsample_nearest2d_backward_out));
    m.impl(
        "upsample_bilinear2d",
        TORCH_FN(torch_nntile::upsample_bilinear2d));
    m.impl(
        "upsample_bilinear2d.out",
        TORCH_FN(torch_nntile::upsample_bilinear2d_out));
    m.impl(
        "upsample_bilinear2d_backward",
        TORCH_FN(torch_nntile::upsample_bilinear2d_backward));
    m.impl(
        "upsample_bilinear2d_backward.grad_input",
        TORCH_FN(torch_nntile::upsample_bilinear2d_backward_out));
}
