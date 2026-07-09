/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_silu_backward.cpp
 */

#include "nntile_executor.h"
#include "nntile_graph_recorder_impl.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <torch/library.h>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

void check_silu_backward(
    const at::Tensor &grad_output,
    const at::Tensor &self)
{
    TORCH_CHECK(
        is_nntile_device(grad_output.device()) &&
            is_nntile_device(self.device()),
        "nntile silu_backward expects nntile tensors");
    TORCH_CHECK(
        grad_output.scalar_type() == at::ScalarType::Float &&
            self.scalar_type() == at::ScalarType::Float,
        "nntile silu_backward supports float32 only");
    TORCH_CHECK(
        grad_output.sizes() == self.sizes(),
        "nntile silu_backward: shape mismatch");
    TORCH_CHECK(
        grad_output.is_contiguous() && self.is_contiguous(),
        "nntile silu_backward requires contiguous tensors");
}

void run_silu_backward(
    const at::Tensor &grad_output,
    const at::Tensor &self,
    at::Tensor &grad_input)
{
    pin_graph_op_inputs({self, grad_output});
    pin_graph_op_output(grad_input, false);
    tensor_silu_backward_fp32(self, grad_output, grad_input);
}

} // namespace

at::Tensor silu_backward(
    const at::Tensor &grad_output,
    const at::Tensor &self)
{
    check_silu_backward(grad_output, self);
    at::Tensor grad_input = at::empty_like(self);
    run_silu_backward(grad_output, self, grad_input);
    return grad_input;
}

at::Tensor &silu_backward_out(
    const at::Tensor &grad_output,
    const at::Tensor &self,
    at::Tensor &grad_input)
{
    check_silu_backward(grad_output, self);
    TORCH_CHECK(
        grad_input.sizes() == self.sizes(),
        "nntile silu_backward.out: output shape mismatch");
    TORCH_CHECK(
        is_nntile_device(grad_input.device()),
        "nntile silu_backward.out: expected nntile output");
    TORCH_CHECK(
        grad_input.is_contiguous(),
        "nntile silu_backward.out requires contiguous out");
    run_silu_backward(grad_output, self, grad_input);
    return grad_input;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("silu_backward", TORCH_FN(torch_nntile::silu_backward));
    m.impl(
        "silu_backward.grad_input",
        TORCH_FN(torch_nntile::silu_backward_out));
}
