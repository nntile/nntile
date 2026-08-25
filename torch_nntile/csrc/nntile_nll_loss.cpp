/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_nll_loss.cpp
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

void check_nll_inputs(
    const at::Tensor &self,
    const at::Tensor &target,
    const std::optional<at::Tensor> &weight)
{
    TORCH_CHECK(
        is_nntile_device(self.device()),
        "nntile nll_loss: self must be on device nntile");
    TORCH_CHECK(
        is_nntile_device(target.device()),
        "nntile nll_loss: target must be on device nntile");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile nll_loss supports float32 only");
    TORCH_CHECK(
        target.scalar_type() == at::ScalarType::Long,
        "nntile nll_loss: target must be int64");
    TORCH_CHECK(
        self.dim() == 2,
        "nntile nll_loss_forward supports 2D log_probs only");
    TORCH_CHECK(
        target.dim() == 1 && target.size(0) == self.size(0),
        "nntile nll_loss: target shape mismatch");
    TORCH_CHECK(
        !weight.has_value() || !weight->defined(),
        "nntile nll_loss: class weight is not supported");
}

} // namespace

std::tuple<at::Tensor, at::Tensor> nll_loss_forward(
    const at::Tensor &self,
    const at::Tensor &target,
    const std::optional<at::Tensor> &weight,
    int64_t reduction,
    int64_t ignore_index)
{
    nntile::GraphFillScope record;
    check_nll_inputs(self, target, weight);
    TORCH_CHECK(
        reduction == at::Reduction::Mean ||
            reduction == at::Reduction::Sum ||
            reduction == at::Reduction::None,
        "nntile nll_loss: unsupported reduction");

    at::Tensor loss;
    if (reduction == at::Reduction::None)
    {
        loss = at::empty(
            {self.size(0)},
            self.options().memory_format(at::MemoryFormat::Contiguous));
    }
    else
    {
        loss = at::empty(
            {},
            self.options().memory_format(at::MemoryFormat::Contiguous));
    }
    at::Tensor total_weight = at::empty(
        {},
        self.options().memory_format(at::MemoryFormat::Contiguous));

    tensor_nll_loss_forward_fp32(
        self,
        target,
        loss,
        total_weight,
        reduction,
        ignore_index);
    return {loss, total_weight};
}

at::Tensor nll_loss_backward(
    const at::Tensor &grad_output,
    const at::Tensor &self,
    const at::Tensor &target,
    const std::optional<at::Tensor> &weight,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor &total_weight)
{
    nntile::GraphFillScope record;
    check_nll_inputs(self, target, weight);
    TORCH_CHECK(
        is_nntile_device(grad_output.device()),
        "nntile nll_loss_backward: grad_output must be on nntile");
    TORCH_CHECK(
        is_nntile_device(total_weight.device()),
        "nntile nll_loss_backward: total_weight must be on nntile");
    TORCH_CHECK(
        grad_output.scalar_type() == at::ScalarType::Float &&
            total_weight.scalar_type() == at::ScalarType::Float,
        "nntile nll_loss_backward supports float32 only");

    at::Tensor grad_input = at::empty_like(self);
    tensor_nll_loss_backward_fp32(
        grad_output,
        self,
        target,
        total_weight,
        grad_input,
        reduction,
        ignore_index);
    return grad_input;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(
        "nll_loss_forward",
        TORCH_FN(torch_nntile::nll_loss_forward));
    m.impl(
        "nll_loss_backward",
        TORCH_FN(torch_nntile::nll_loss_backward));
}
