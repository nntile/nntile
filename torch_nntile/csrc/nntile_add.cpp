/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_add.cpp
 * Out-of-place aten::add for device=nntile (torch-native StarPU path).
 */

#include "nntile_executor.h"

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
    const at::Tensor &other)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) &&
            is_nntile_device(other.device()),
        "nntile add expects both operands on device nntile");
    TORCH_CHECK(
        self.sizes().equals(other.sizes()),
        "nntile torch_add: same-shape tensors only "
        "(broadcast disabled under NNTILE_TORCH_NATIVE_OPS)");
    TORCH_CHECK(
        self.scalar_type() == other.scalar_type(),
        "nntile add: dtype mismatch");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile torch_add supports float32 only");
    TORCH_CHECK(
        self.is_contiguous() && other.is_contiguous(),
        "nntile add requires contiguous tensors");
}

//! Probe output meta via device=meta (shared Torch shape logic).
at::Tensor meta_add_result(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha)
{
    auto self_m = at::empty_like(
        self,
        self.options().device(at::kMeta));
    auto other_m = at::empty_like(
        other,
        other.options().device(at::kMeta));
    return at::add(self_m, other_m, alpha);
}

void run_torch_add(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha,
    at::Tensor &out)
{
    // tensor_add_fp32(alpha_x, x, beta_y, y, out) with alpha_x=1,
    // beta_y = torch alpha → out = x + alpha * y.
    tensor_add_fp32(
        1.0f,
        self,
        alpha.to<float>(),
        other,
        out);
}

} // namespace

at::Tensor add_tensor(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha)
{
    check_add_inputs(self, other);
    at::Tensor out_meta = meta_add_result(self, other, alpha);
    at::Tensor out = at::empty(
        out_meta.sizes(),
        self.options().memory_format(at::MemoryFormat::Contiguous));
    run_torch_add(self, other, alpha, out);
    return out;
}

at::Tensor &add_out(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha,
    at::Tensor &out)
{
    check_add_inputs(self, other);
    TORCH_CHECK(
        is_nntile_device(out.device()),
        "nntile add.out expects output on device nntile");
    TORCH_CHECK(
        out.sizes().equals(self.sizes()),
        "nntile add.out: output shape mismatch");
    TORCH_CHECK(
        out.scalar_type() == at::ScalarType::Float &&
            out.is_contiguous(),
        "nntile add.out requires contiguous float32 output");
    run_torch_add(self, other, alpha, out);
    return out;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("add.Tensor", TORCH_FN(torch_nntile::add_tensor));
    m.impl("add.out", TORCH_FN(torch_nntile::add_out));
}
