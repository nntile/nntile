/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_gelu_backward.cpp
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

bool is_gelu_tanh_approximate(c10::string_view approximate)
{
    return approximate == "tanh";
}

void check_gelu_backward(
    const at::Tensor &grad_output,
    const at::Tensor &self,
    c10::string_view approximate)
{
    TORCH_CHECK(
        is_nntile_device(grad_output.device()) &&
            is_nntile_device(self.device()),
        "nntile gelu_backward expects nntile tensors");
    TORCH_CHECK(
        approximate == "none" || approximate == "tanh",
        "nntile gelu_backward supports approximate='none' or 'tanh'");
    TORCH_CHECK(
        grad_output.scalar_type() == at::ScalarType::Float &&
            self.scalar_type() == at::ScalarType::Float,
        "nntile gelu_backward supports float32 only");
    TORCH_CHECK(
        grad_output.sizes() == self.sizes(),
        "nntile gelu_backward: shape mismatch");
}

void run_gelu_backward(
    const at::Tensor &grad_output,
    const at::Tensor &self,
    at::Tensor &grad_input,
    bool approximate_tanh)
{
    tensor_gelu_backward_fp32(
        self,
        grad_output,
        grad_input,
        approximate_tanh);
}

} // namespace

at::Tensor gelu_backward(
    const at::Tensor &grad_output,
    const at::Tensor &self,
    c10::string_view approximate)
{
    nntile::GraphFillScope record;
    check_gelu_backward(grad_output, self, approximate);
    at::Tensor grad_input = at::empty_like(self);
    run_gelu_backward(
        grad_output,
        self,
        grad_input,
        is_gelu_tanh_approximate(approximate));
    return grad_input;
}

at::Tensor &gelu_backward_out(
    const at::Tensor &grad_output,
    const at::Tensor &self,
    c10::string_view approximate,
    at::Tensor &grad_input)
{
    nntile::GraphFillScope record;
    check_gelu_backward(grad_output, self, approximate);
    TORCH_CHECK(
        grad_input.sizes() == self.sizes(),
        "nntile gelu_backward.out: output shape mismatch");
    TORCH_CHECK(
        is_nntile_device(grad_input.device()),
        "nntile gelu_backward.out: expected nntile output");
    run_gelu_backward(
        grad_output,
        self,
        grad_input,
        is_gelu_tanh_approximate(approximate));
    return grad_input;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("gelu_backward", TORCH_FN(torch_nntile::gelu_backward));
    m.impl(
        "gelu_backward.grad_input",
        TORCH_FN(torch_nntile::gelu_backward_out));
}
