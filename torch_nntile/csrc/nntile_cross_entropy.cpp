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

#include <chrono>
#include <vector>

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

std::vector<int64_t> maxsumexp_pytorch_shape(c10::IntArrayRef logits_sizes)
{
    std::vector<int64_t> shape;
    shape.reserve(static_cast<std::size_t>(logits_sizes.size()));
    const int64_t class_axis = logits_sizes.size() - 1;
    for (int64_t i = 0; i < logits_sizes.size(); ++i)
    {
        if (i != class_axis)
        {
            shape.push_back(logits_sizes[i]);
        }
    }
    shape.push_back(2);
    return shape;
}

} // namespace

std::tuple<at::Tensor, at::Tensor> cross_entropy_forward(
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
    at::Tensor maxsumexp = empty_metadata_tensor(
        maxsumexp_pytorch_shape(logits.sizes()),
        at::kFloat,
        logits.device());
#ifndef TORCH_NNTILE_USE_LIBNNTILE
    ensure_host_staging(loss);
    ensure_host_staging(maxsumexp);
#endif
    tensor_cross_entropy_forward_fp32(
        logits,
        target,
        ignore_index,
        reduction_is_mean(reduction),
        loss,
        maxsumexp);
    return {loss, maxsumexp};
}

at::Tensor cross_entropy_backward(
    const at::Tensor &logits,
    const at::Tensor &target,
    const at::Tensor &grad_output,
    const at::Tensor &maxsumexp,
    int64_t reduction,
    int64_t ignore_index)
{
    const auto t0 = std::chrono::steady_clock::now();
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
    TORCH_CHECK(
        is_nntile_device(maxsumexp.device()),
        "nntile cross_entropy_backward: maxsumexp must be on device nntile");
    at::Tensor grad_out = grad_output;
#ifndef TORCH_NNTILE_USE_LIBNNTILE
    ensure_host_staging(grad_out);
    if (grad_out.numel() == 1)
    {
        grad_out.fill_(1.0f);
    }
#endif
    at::Tensor grad_logits = empty_metadata_tensor(
        logits.sizes(),
        logits.scalar_type(),
        logits.device());
    at::Tensor grad_row = empty_metadata_tensor(
        target.sizes(),
        logits.scalar_type(),
        logits.device());
    tensor_cross_entropy_backward_fp32(
        logits,
        target,
        grad_out,
        maxsumexp,
        grad_row,
        grad_logits,
        ignore_index,
        reduction_is_mean(reduction));
#ifdef TORCH_NNTILE_USE_LIBNNTILE
    note_record_ce_bwd(
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0)
            .count());
#endif
    return grad_logits;
}

} // namespace torch_nntile
