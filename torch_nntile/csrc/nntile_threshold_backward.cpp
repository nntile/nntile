/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_threshold_backward.cpp
 */

#include "nntile_executor.h"
#include "nntile_graph_recorder_impl.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <torch/library.h>

#include <chrono>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

void check_threshold_backward(
    const at::Tensor &grad_output,
    const at::Tensor &self)
{
    TORCH_CHECK(
        is_nntile_device(grad_output.device()) &&
            is_nntile_device(self.device()),
        "nntile threshold_backward expects nntile tensors");
    TORCH_CHECK(
        grad_output.scalar_type() == at::ScalarType::Float &&
            self.scalar_type() == at::ScalarType::Float,
        "nntile threshold_backward supports float32 only");
    TORCH_CHECK(
        grad_output.sizes() == self.sizes(),
        "nntile threshold_backward: shape mismatch grad_output=",
        grad_output.sizes(),
        " self=",
        self.sizes());
    TORCH_CHECK(
        grad_output.is_contiguous() && self.is_contiguous(),
        "nntile threshold_backward requires contiguous tensors");
}

} // namespace

at::Tensor threshold_backward(
    const at::Tensor &grad_output,
    const at::Tensor &self,
    const at::Scalar &threshold)
{
    const auto t0 = std::chrono::steady_clock::now();
    check_threshold_backward(grad_output, self);
    TORCH_CHECK(
        threshold.to<double>() == 0.0,
        "nntile threshold_backward supports ReLU only (threshold=0)");
    at::Tensor grad_input = at::empty_like(self);
    pin_graph_op_inputs({self, grad_output});
    pin_graph_op_output(grad_input, false);
    tensor_relu_backward_fp32(self, grad_output, grad_input);
#ifdef TORCH_NNTILE_USE_LIBNNTILE
    note_record_relu_bwd(
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0)
            .count());
#endif
    return grad_input;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(
        "threshold_backward",
        TORCH_FN(torch_nntile::threshold_backward));
}
