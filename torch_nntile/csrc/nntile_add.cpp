/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_add.cpp
 */

#include "nntile_executor.h"
#include "nntile_graph_recorder_impl.h"

#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

void check_add_inputs(
    const at::Tensor &self,
    const at::Tensor &other,
    const std::optional<at::Tensor> &out = std::nullopt)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) &&
            is_nntile_device(other.device()),
        "nntile add expects both operands on device nntile");
    if (out.has_value())
    {
        TORCH_CHECK(
            is_nntile_device(out->device()),
            "nntile add.out expects output on device nntile");
    }
    TORCH_CHECK(self.sizes() == other.sizes(), "nntile add: shape mismatch");
    TORCH_CHECK(
        self.scalar_type() == other.scalar_type(),
        "nntile add: dtype mismatch");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile add supports float32 only in phase 2");
    TORCH_CHECK(
        self.is_contiguous() && other.is_contiguous(),
        "nntile add requires contiguous tensors");
    if (out.has_value())
    {
        TORCH_CHECK(
            out->sizes() == self.sizes(),
            "nntile add.out: output shape mismatch");
        TORCH_CHECK(
            out->is_contiguous(),
            "nntile add.out requires contiguous output");
    }
}

void check_add_inplace_inputs(
    const at::Tensor &self,
    const at::Tensor &other)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) &&
            is_nntile_device(other.device()),
        "nntile add expects both operands on device nntile");
    TORCH_CHECK(
        at::are_expandable(other.sizes(), self.sizes()),
        "nntile add_.Tensor: shape not broadcastable to self");
    TORCH_CHECK(
        self.scalar_type() == other.scalar_type(),
        "nntile add: dtype mismatch");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile add supports float32 only in phase 2");
    TORCH_CHECK(
        self.is_contiguous() && other.is_contiguous(),
        "nntile add requires contiguous tensors");
}

at::Tensor broadcast_to_self_shape(
    const at::Tensor &tensor,
    c10::IntArrayRef self_sizes)
{
    if (tensor.sizes().equals(self_sizes))
    {
        return tensor;
    }
    at::Tensor expanded = tensor.expand(self_sizes);
    if (!expanded.is_contiguous())
    {
        expanded = expanded.contiguous();
    }
    return expanded;
}

void run_add_kernel(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha,
    at::Tensor &out)
{
    const float self_scale = 1.0f;
    const float other_scale = alpha.to<float>();
    pin_graph_op_inputs({self, other});
    pin_graph_op_output(out, true);
    tensor_add_fp32(
        self_scale,
        self.data_ptr<float>(),
        other_scale,
        other.data_ptr<float>(),
        out.data_ptr<float>(),
        self.sizes());
}

} // namespace

at::Tensor add_tensor(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha)
{
    check_add_inputs(self, other);
    at::Tensor out = at::empty_like(self);
    run_add_kernel(self, other, alpha, out);
    return out;
}

at::Tensor &add_out(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha,
    at::Tensor &out)
{
    check_add_inputs(self, other, out);
    run_add_kernel(self, other, alpha, out);
    return out;
}

at::Tensor &add_inplace_tensor(
    at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha)
{
    check_add_inplace_inputs(self, other);
    const at::Tensor other_broadcast =
        broadcast_to_self_shape(other, self.sizes());
    const float other_scale = alpha.to<float>();
    const float self_scale = 1.0f;
    pin_graph_op_inputs({self, other_broadcast});
    pin_graph_op_output(self, true);
    tensor_add_inplace_fp32(
        other_scale,
        other_broadcast.data_ptr<float>(),
        self_scale,
        self.data_ptr<float>(),
        self.sizes());
    return self;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("add.Tensor", TORCH_FN(torch_nntile::add_tensor));
    m.impl("add.out", TORCH_FN(torch_nntile::add_out));
    m.impl("add_.Tensor", TORCH_FN(torch_nntile::add_inplace_tensor));
}
