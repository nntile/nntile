/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_cross_entropy.cpp
 */

#include "nntile_executor.h"
#include "nntile_graph_recorder_impl.h"
#include "nntile_tensor_gc.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

void check_cross_entropy_inputs(
    const at::Tensor &logits,
    const at::Tensor &target)
{
    TORCH_CHECK(
        is_nntile_device(logits.device()),
        "nntile cross_entropy: logits must be on device nntile");
    TORCH_CHECK(
        target.scalar_type() == at::ScalarType::Long,
        "nntile cross_entropy: target must be int64");
    TORCH_CHECK(
        is_nntile_device(target.device()),
        "nntile cross_entropy: target must be on device nntile");
    TORCH_CHECK(logits.dim() >= 2, "nntile cross_entropy: logits must be >= 2D");
    TORCH_CHECK(
        target.dim() + 1 == logits.dim(),
        "nntile cross_entropy: target shape must match logits without class dim");
    TORCH_CHECK(
        logits.scalar_type() == at::ScalarType::Float,
        "nntile cross_entropy supports float32 logits only");
    TORCH_CHECK(
        logits.is_contiguous() && target.is_contiguous(),
        "nntile cross_entropy requires contiguous tensors");
    for (int64_t i = 0; i < target.dim(); ++i)
    {
        TORCH_CHECK(
            target.size(i) == logits.size(i),
            "nntile cross_entropy: batch shape mismatch");
    }
}

bool reduction_is_mean(int64_t reduction)
{
    // Matches torch.nn._reduction: mean=1, sum=2, none=0
    return reduction == 1;
}

} // namespace

at::Tensor cross_entropy_forward(
    const at::Tensor &logits,
    const at::Tensor &target,
    int64_t reduction,
    int64_t ignore_index)
{
    check_cross_entropy_inputs(logits, target);
    TORCH_CHECK(
        reduction == 1 || reduction == 2,
        "nntile cross_entropy supports reduction mean (1) or sum (2) only");

    at::Tensor loss = empty_metadata_tensor({}, at::kFloat, logits.device());
#ifdef TORCH_NNTILE_USE_LIBNNTILE
#else
    ensure_host_staging(loss);
#endif
    pin_graph_op_inputs({logits, target});
    pin_graph_op_output(loss, true);
    tensor_cross_entropy_forward_fp32(
        logits,
        target,
        ignore_index,
        reduction_is_mean(reduction),
        loss);
    return loss;
}

at::Tensor cross_entropy_backward(
    const at::Tensor &logits,
    const at::Tensor &target,
    const at::Tensor &grad_output,
    int64_t reduction,
    int64_t ignore_index)
{
    check_cross_entropy_inputs(logits, target);
    TORCH_CHECK(
        reduction == 1 || reduction == 2,
        "nntile cross_entropy supports reduction mean (1) or sum (2) only");
    TORCH_CHECK(
        grad_output.numel() == 1,
        "nntile cross_entropy_backward expects scalar grad_output");
    TORCH_CHECK(
        is_nntile_device(grad_output.device()),
        "nntile cross_entropy_backward: grad_output must be on device nntile");
    at::Tensor grad_out = grad_output;
#ifdef TORCH_NNTILE_USE_LIBNNTILE
    if (!is_metadata_only_tensor(grad_out))
    {
        ensure_host_staging(grad_out);
    }
#else
    if (!has_host_staging(grad_out))
    {
        ensure_host_staging(grad_out);
        if (grad_out.numel() == 1)
        {
            grad_out.fill_(1.0f);
        }
    }
    if (has_host_staging(grad_out))
    {
        mark_staged_input_tensor(grad_out);
    }
#endif
    at::Tensor grad_logits = at::empty_like(logits);
    at::Tensor grad_row = at::empty(target.sizes(), logits.options());
    pin_graph_op_inputs({logits, target, grad_out});
    pin_graph_op_output(grad_logits, false);
    tensor_cross_entropy_backward_fp32(
        logits,
        target,
        grad_out,
        grad_row,
        grad_logits,
        ignore_index,
        reduction_is_mean(reduction));
    return grad_logits;
}

} // namespace torch_nntile
