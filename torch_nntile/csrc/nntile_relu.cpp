/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_relu.cpp
 */

#include "nntile_executor.h"

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

void check_relu_input(
    const at::Tensor &self,
    const std::optional<at::Tensor> &out = std::nullopt)
{
    TORCH_CHECK(is_nntile_device(self.device()), "nntile relu: expected nntile");
    if (out.has_value())
    {
        TORCH_CHECK(
            is_nntile_device(out->device()),
            "nntile relu.out: expected nntile output");
    }
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile relu supports float32 only");
    if (out.has_value())
    {
        TORCH_CHECK(
            out->sizes() == self.sizes(),
            "nntile relu.out: output shape mismatch");
    }
}

void run_relu(const at::Tensor &self, at::Tensor &out)
{
    // Lifetimes: PyTorch ReluBackward0 saves the output; do not pin ``self``.
    tensor_relu_fp32(self, out);
}

} // namespace

at::Tensor relu(const at::Tensor &self)
{
    nntile::GraphFillScope record;
    check_relu_input(self);
    at::Tensor out = at::empty_like(self);
    run_relu(self, out);
    return out;
}

at::Tensor &relu_out(const at::Tensor &self, at::Tensor &out)
{
    nntile::GraphFillScope record;
    check_relu_input(self, out);
    run_relu(self, out);
    return out;
}

at::Tensor &relu_(at::Tensor &self)
{
    nntile::GraphFillScope record;
    // Functional SSA rebind (same pattern as gelu_ / silu_).
    check_relu_input(self, self);
    run_relu(self, self);
    return self;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("relu", TORCH_FN(torch_nntile::relu));
    m.impl("relu.out", TORCH_FN(torch_nntile::relu_out));
    m.impl("relu_", TORCH_FN(torch_nntile::relu_));
}
