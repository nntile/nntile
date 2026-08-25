/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_adaptive_avg_pool2d.cpp
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

void check_input(const at::Tensor &input, const char *name)
{
    TORCH_CHECK(is_nntile_device(input.device()), name, ": expected nntile");
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Float, name, ": fp32");
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

} // namespace

at::Tensor adaptive_avg_pool2d(
    const at::Tensor &input,
    c10::SymIntArrayRef output_size)
{
    nntile::GraphFillScope record;
    check_input(input, "nntile _adaptive_avg_pool2d");
    std::vector<int64_t> out_shape(input.sizes().begin(), input.sizes().end());
    TORCH_CHECK(out_shape.size() >= 2, "adaptive_avg_pool2d: rank < 2");
    std::vector<int64_t> out_hw = sym_to_i64(output_size);
    TORCH_CHECK(out_hw.size() == 2, "adaptive_avg_pool2d: expected 2D");
    out_shape[out_shape.size() - 2] = out_hw[0];
    out_shape[out_shape.size() - 1] = out_hw[1];
    at::Tensor out = empty_metadata_tensor(
        out_shape,
        input.scalar_type(),
        input.device());
    tensor_adaptive_avg_pool2d_fp32(input, out, out_hw);
    return out;
}

at::Tensor &adaptive_avg_pool2d_out(
    const at::Tensor &input,
    c10::SymIntArrayRef output_size,
    at::Tensor &out)
{
    nntile::GraphFillScope record;
    check_input(input, "nntile _adaptive_avg_pool2d.out");
    TORCH_CHECK(is_nntile_device(out.device()), "adaptive out: nntile out");
    std::vector<int64_t> out_hw = sym_to_i64(output_size);
    tensor_adaptive_avg_pool2d_fp32(input, out, out_hw);
    return out;
}

at::Tensor adaptive_avg_pool2d_backward(
    const at::Tensor &grad_output,
    const at::Tensor &input)
{
    nntile::GraphFillScope record;
    check_input(input, "nntile _adaptive_avg_pool2d_backward");
    check_input(grad_output, "nntile _adaptive_avg_pool2d_backward grad");
    at::Tensor grad_input = empty_metadata_tensor(
        input.sizes(),
        input.scalar_type(),
        input.device());
    tensor_adaptive_avg_pool2d_backward_fp32(
        grad_output,
        input,
        grad_input);
    return grad_input;
}

at::Tensor &adaptive_avg_pool2d_backward_out(
    const at::Tensor &grad_output,
    const at::Tensor &input,
    at::Tensor &grad_input)
{
    nntile::GraphFillScope record;
    check_input(input, "nntile _adaptive_avg_pool2d_backward.out input");
    check_input(grad_output, "nntile _adaptive_avg_pool2d_backward.out grad");
    tensor_adaptive_avg_pool2d_backward_fp32(
        grad_output,
        input,
        grad_input);
    return grad_input;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(
        "_adaptive_avg_pool2d",
        TORCH_FN(torch_nntile::adaptive_avg_pool2d));
    m.impl(
        "_adaptive_avg_pool2d.out",
        TORCH_FN(torch_nntile::adaptive_avg_pool2d_out));
    m.impl(
        "_adaptive_avg_pool2d_backward",
        TORCH_FN(torch_nntile::adaptive_avg_pool2d_backward));
    m.impl(
        "_adaptive_avg_pool2d_backward.out",
        TORCH_FN(torch_nntile::adaptive_avg_pool2d_backward_out));
}
