/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_hypot.cpp
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

void check_hypot_inputs(
    const at::Tensor &self,
    const at::Tensor &other,
    const std::optional<at::Tensor> &out = std::nullopt)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) &&
            is_nntile_device(other.device()),
        "nntile hypot expects both operands on device nntile");
    if (out.has_value())
    {
        TORCH_CHECK(
            is_nntile_device(out->device()),
            "nntile hypot.out expects output on device nntile");
    }
    TORCH_CHECK(
        self.sizes() == other.sizes(),
        "nntile hypot: shape mismatch");
    TORCH_CHECK(
        self.scalar_type() == other.scalar_type(),
        "nntile hypot: dtype mismatch");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile hypot supports float32 only in phase 2");
    if (out.has_value())
    {
        TORCH_CHECK(
            out->sizes() == self.sizes(),
            "nntile hypot.out: output shape mismatch");
    }
}

void run_hypot_kernel(
    const at::Tensor &self,
    const at::Tensor &other,
    at::Tensor &out)
{
    tensor_hypot_fp32(self, other, out);
}

} // namespace

at::Tensor hypot_tensor(const at::Tensor &self, const at::Tensor &other)
{
    nntile::GraphFillScope record;
    check_hypot_inputs(self, other);
    at::Tensor out = at::empty_like(self);
    run_hypot_kernel(self, other, out);
    return out;
}

at::Tensor &hypot_out(
    const at::Tensor &self,
    const at::Tensor &other,
    at::Tensor &out)
{
    nntile::GraphFillScope record;
    check_hypot_inputs(self, other, out);
    run_hypot_kernel(self, other, out);
    return out;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("hypot", TORCH_FN(torch_nntile::hypot_tensor));
    m.impl("hypot.out", TORCH_FN(torch_nntile::hypot_out));
}
