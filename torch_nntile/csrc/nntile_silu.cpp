/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_silu.cpp
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

void check_silu_input(
    const at::Tensor &self,
    const std::optional<at::Tensor> &out = std::nullopt)
{
    TORCH_CHECK(is_nntile_device(self.device()), "nntile silu: expected nntile");
    if (out.has_value())
    {
        TORCH_CHECK(
            is_nntile_device(out->device()),
            "nntile silu.out: expected nntile output");
    }
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile silu supports float32 only");
    TORCH_CHECK(self.is_contiguous(), "nntile silu requires contiguous input");
    if (out.has_value())
    {
        TORCH_CHECK(
            out->sizes() == self.sizes(),
            "nntile silu.out: output shape mismatch");
        TORCH_CHECK(
            out->is_contiguous(),
            "nntile silu.out requires contiguous out");
    }
}

void run_silu(const at::Tensor &self, at::Tensor &out)
{
    pin_graph_op_inputs({self});
    pin_graph_op_output(out, true);
    tensor_silu_fp32(
        self.data_ptr<float>(),
        out.data_ptr<float>(),
        self.sizes());
}

} // namespace

at::Tensor silu(const at::Tensor &self)
{
    check_silu_input(self);
    at::Tensor out = at::empty_like(self);
    run_silu(self, out);
    return out;
}

at::Tensor &silu_out(const at::Tensor &self, at::Tensor &out)
{
    check_silu_input(self, out);
    run_silu(self, out);
    return out;
}

at::Tensor &silu_(at::Tensor &self)
{
    check_silu_input(self, self);
    run_silu(self, self);
    return self;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("silu", TORCH_FN(torch_nntile::silu));
    m.impl("silu.out", TORCH_FN(torch_nntile::silu_out));
    m.impl("silu_", TORCH_FN(torch_nntile::silu_));
}
