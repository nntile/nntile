/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_gelu.cpp
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

void check_gelu_input(
    const at::Tensor &self,
    c10::string_view approximate,
    const std::optional<at::Tensor> &out = std::nullopt)
{
    TORCH_CHECK(is_nntile_device(self.device()), "nntile gelu: expected nntile");
    if (out.has_value())
    {
        TORCH_CHECK(
            is_nntile_device(out->device()),
            "nntile gelu.out: expected nntile output");
    }
    TORCH_CHECK(
        approximate == "none" || approximate == "tanh",
        "nntile gelu supports approximate='none' or 'tanh'");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile gelu supports float32 only");
    TORCH_CHECK(self.is_contiguous(), "nntile gelu requires contiguous input");
    if (out.has_value())
    {
        TORCH_CHECK(
            out->sizes() == self.sizes(),
            "nntile gelu.out: output shape mismatch");
        TORCH_CHECK(
            out->is_contiguous(),
            "nntile gelu.out requires contiguous out");
    }
}

void run_gelu(
    const at::Tensor &self,
    at::Tensor &out,
    bool approximate_tanh)
{
    if (self.is_same(out))
    {
        tensor_gelu_inplace_fp32(out, approximate_tanh);
        return;
    }
    tensor_gelu_fp32(self, out, approximate_tanh);
}

} // namespace

at::Tensor gelu(
    const at::Tensor &self,
    c10::string_view approximate)
{
    check_gelu_input(self, approximate);
    at::Tensor out = at::empty_like(self);
    run_gelu(self, out, is_gelu_tanh_approximate(approximate));
    return out;
}

at::Tensor &gelu_out(
    const at::Tensor &self,
    c10::string_view approximate,
    at::Tensor &out)
{
    check_gelu_input(self, approximate, out);
    run_gelu(self, out, is_gelu_tanh_approximate(approximate));
    return out;
}

at::Tensor &gelu_(at::Tensor &self, c10::string_view approximate)
{
    check_gelu_input(self, approximate, self);
    run_gelu(self, self, is_gelu_tanh_approximate(approximate));
    return self;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("gelu", TORCH_FN(torch_nntile::gelu));
    m.impl("gelu.out", TORCH_FN(torch_nntile::gelu_out));
    m.impl("gelu_", TORCH_FN(torch_nntile::gelu_));
}
