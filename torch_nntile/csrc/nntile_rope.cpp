/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_rope.cpp
 */

#include "nntile_rope.h"

#include "nntile_executor_classic.h"
#include "nntile_graph_recorder_impl.h"

#include <ATen/TensorUtils.h>

#include <array>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

void check_rope_tensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(
        is_nntile_device(tensor.device()),
        "nntile rope: expected nntile ",
        name);
    TORCH_CHECK(
        tensor.scalar_type() == at::ScalarType::Float,
        "nntile rope supports float32 only");
    TORCH_CHECK(tensor.is_contiguous(), "nntile rope requires contiguous");
}

void check_rope_shapes(
    const at::Tensor &sin,
    const at::Tensor &cos,
    const at::Tensor &x)
{
    TORCH_CHECK(
        sin.sizes().equals(cos.sizes()),
        "nntile rope: sin and cos shapes must match");
    TORCH_CHECK(sin.dim() >= 1, "nntile rope: sin/cos must be at least 1D");
    TORCH_CHECK(
        x.dim() >= sin.dim(),
        "nntile rope: x rank must be >= sin rank");
    const int64_t axis_shift = x.dim() - sin.dim();
    const int64_t half_axis = sin.dim() - 1;
    for (int64_t i = 0; i < half_axis; ++i)
    {
        TORCH_CHECK(
            x.size(i + axis_shift) == sin.size(i),
            "nntile rope: x/sin batch axis mismatch at ",
            i);
    }
    TORCH_CHECK(
        x.size(half_axis + axis_shift) == 2 * sin.size(half_axis),
        "nntile rope: x head axis must be 2 * sin half axis");
}

} // namespace

at::Tensor rope_forward(
    const at::Tensor &sin,
    const at::Tensor &cos,
    const at::Tensor &x)
{
    nntile::GraphFillScope record;
    check_rope_tensor(sin, "sin");
    check_rope_tensor(cos, "cos");
    check_rope_tensor(x, "x");
    check_rope_shapes(sin, cos, x);

    at::Tensor out = at::empty(
        x.sizes(),
        x.options().memory_format(at::MemoryFormat::Contiguous));
    classic_tensor_rope_fp32(sin, cos, x, out);
    return out;
}

at::Tensor rope_backward(
    const at::Tensor &sin,
    const at::Tensor &cos,
    const at::Tensor &grad_out,
    std::array<bool, 1> output_mask)
{
    nntile::GraphFillScope record;
    check_rope_tensor(sin, "sin");
    check_rope_tensor(cos, "cos");
    check_rope_tensor(grad_out, "grad_out");
    check_rope_shapes(sin, cos, grad_out);

    at::Tensor grad_x;
    if (output_mask[0])
    {
        grad_x = at::empty(
            grad_out.sizes(),
            grad_out.options().memory_format(
                at::MemoryFormat::Contiguous));
        classic_tensor_rope_backward_fp32(sin, cos, grad_out, grad_x);
    }
    return grad_x;
}

} // namespace torch_nntile
