/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_add_fiber.cpp
 */

#include "nntile_add_fiber.h"

#include "nntile_executor.h"
#include "nntile_graph_recorder_impl.h"
#include "nntile_tensor_gc.h"

#include <ATen/TensorUtils.h>

#include <array>
#include <vector>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

void check_add_fiber_inputs(
    const at::Tensor &fiber,
    const at::Tensor &tensor,
    int64_t axis,
    int64_t batch_ndim)
{
    TORCH_CHECK(
        is_nntile_device(fiber.device()) && is_nntile_device(tensor.device()),
        "nntile add_fiber expects nntile tensors");
    TORCH_CHECK(
        fiber.scalar_type() == at::ScalarType::Float &&
            tensor.scalar_type() == at::ScalarType::Float,
        "nntile add_fiber supports float32 only");
    TORCH_CHECK(
        fiber.is_contiguous() && tensor.is_contiguous(),
        "nntile add_fiber requires contiguous tensors");
    TORCH_CHECK(
        fiber.dim() == batch_ndim + 1,
        "nntile add_fiber: fiber ndim must be batch_ndim + 1");
    TORCH_CHECK(
        axis >= 0 && axis < tensor.dim(),
        "nntile add_fiber: axis out of range");
    TORCH_CHECK(
        batch_ndim >= 0 && batch_ndim <= tensor.dim(),
        "nntile add_fiber: batch_ndim out of range");
    for (int64_t i = 0; i < batch_ndim; ++i)
    {
        TORCH_CHECK(
            fiber.size(i) == tensor.size(i),
            "nntile add_fiber: fiber batch dim ",
            i,
            " must match tensor");
    }
    TORCH_CHECK(
        fiber.size(batch_ndim) == tensor.size(axis),
        "nntile add_fiber: fiber size must match tensor axis");
}

} // namespace

at::Tensor add_fiber_forward(
    const at::Tensor &fiber,
    const at::Tensor &tensor,
    int64_t axis,
    int64_t batch_ndim,
    double alpha,
    double beta)
{
    check_add_fiber_inputs(fiber, tensor, axis, batch_ndim);
    at::Tensor out = at::empty(
        tensor.sizes(),
        tensor.options().memory_format(at::MemoryFormat::Contiguous));
    tensor_add_fiber_fp32(
        static_cast<float>(alpha),
        fiber,
        static_cast<float>(beta),
        tensor,
        out,
        axis,
        batch_ndim);
    return out;
}

std::tuple<at::Tensor, at::Tensor> add_fiber_backward(
    const at::Tensor &grad_out,
    const at::Tensor &fiber,
    const at::Tensor &tensor,
    int64_t axis,
    int64_t batch_ndim,
    std::array<bool, 2> output_mask,
    double alpha,
    double beta)
{
    TORCH_CHECK(
        is_nntile_device(grad_out.device()),
        "nntile add_fiber_backward expects nntile grad_out");
    TORCH_CHECK(
        grad_out.scalar_type() == at::ScalarType::Float,
        "nntile add_fiber_backward supports float32 only");
    TORCH_CHECK(
        grad_out.is_contiguous(),
        "nntile add_fiber_backward requires contiguous grad_out");
    check_add_fiber_inputs(fiber, tensor, axis, batch_ndim);
    TORCH_CHECK(
        grad_out.sizes().equals(tensor.sizes()),
        "nntile add_fiber_backward: grad_out shape must match tensor");
    TORCH_CHECK(
        beta == 1.0,
        "nntile add_fiber_backward currently supports beta=1 only");

    at::Tensor grad_fiber;
    at::Tensor grad_tensor;
    if (output_mask[0])
    {
        grad_fiber = at::empty(
            fiber.sizes(),
            fiber.options().memory_format(at::MemoryFormat::Contiguous));
        tensor_sum_fiber_fp32(
            grad_out,
            grad_fiber,
            axis,
            batch_ndim,
            static_cast<float>(alpha));
#ifdef TORCH_NNTILE_USE_LIBNNTILE
        std::vector<nntile::Index> fiber_shape;
        fiber_shape.reserve(static_cast<std::size_t>(grad_fiber.dim()));
        for (const auto dim : grad_fiber.sizes())
        {
            fiber_shape.push_back(static_cast<nntile::Index>(dim));
        }
        nntile::TensorGraph::TensorNode *grad_fiber_node =
            lookup_data_node(grad_fiber, fiber_shape);
        if (grad_fiber_node != nullptr)
        {
            register_param_grad_node(fiber, grad_fiber_node);
            at::Tensor grad_fiber_alias = grad_fiber;
            register_grad_alias_for_host_copy(
                grad_fiber_alias, grad_fiber_node);
        }
#endif
    }
    if (output_mask[1])
    {
        grad_tensor = grad_out;
#ifdef TORCH_NNTILE_USE_LIBNNTILE
        if (tensor.defined())
        {
            std::vector<nntile::Index> tensor_shape;
            tensor_shape.reserve(static_cast<std::size_t>(grad_tensor.dim()));
            for (const auto dim : grad_tensor.sizes())
            {
                tensor_shape.push_back(static_cast<nntile::Index>(dim));
            }
            nntile::TensorGraph::TensorNode *grad_tensor_node =
                lookup_data_node(grad_tensor, tensor_shape);
            if (grad_tensor_node != nullptr)
            {
                register_param_grad_node(tensor, grad_tensor_node);
                at::Tensor grad_tensor_alias = grad_tensor;
                register_grad_alias_for_host_copy(
                    grad_tensor_alias, grad_tensor_node);
            }
        }
#endif
    }
    return {grad_fiber, grad_tensor};
}

} // namespace torch_nntile
