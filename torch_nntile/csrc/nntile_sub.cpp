/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_sub.cpp
 * Out-of-place aten::sub for device=nntile (torch-native StarPU path).
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

void check_sub_inputs(
    const at::Tensor &self,
    const at::Tensor &other)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) &&
            is_nntile_device(other.device()),
        "nntile sub expects both operands on device nntile");
    TORCH_CHECK(
        self.sizes().equals(other.sizes()),
        "nntile torch_sub: same-shape tensors only");
    TORCH_CHECK(
        self.scalar_type() == other.scalar_type(),
        "nntile sub: dtype mismatch");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile torch_sub supports float32 only");
    TORCH_CHECK(
        self.is_contiguous() && other.is_contiguous(),
        "nntile sub requires contiguous tensors");
}

void run_torch_sub(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha,
    at::Tensor &out)
{
    tensor_sub_fp32(self, other, alpha.to<float>(), out);
}

} // namespace

at::Tensor sub_tensor(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha)
{
    check_sub_inputs(self, other);
    at::Tensor out = at::empty_like(self);
    run_torch_sub(self, other, alpha, out);
    return out;
}

at::Tensor &sub_out(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha,
    at::Tensor &out)
{
    check_sub_inputs(self, other);
    TORCH_CHECK(
        is_nntile_device(out.device()),
        "nntile sub.out expects output on device nntile");
    TORCH_CHECK(
        out.sizes().equals(self.sizes()),
        "nntile sub.out: output shape mismatch");
    TORCH_CHECK(
        out.scalar_type() == at::ScalarType::Float &&
            out.is_contiguous(),
        "nntile sub.out requires contiguous float32 output");
    run_torch_sub(self, other, alpha, out);
    return out;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("sub.Tensor", TORCH_FN(torch_nntile::sub_tensor));
    m.impl("sub.out", TORCH_FN(torch_nntile::sub_out));
}
