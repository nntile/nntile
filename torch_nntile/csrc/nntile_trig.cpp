/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_trig.cpp
 * Torch-native cos / sin / neg / rsqrt (StarPU unary family).
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

void check_unary_fp32(
    const at::Tensor &self,
    const char *name,
    const std::optional<at::Tensor> &out = std::nullopt)
{
    TORCH_CHECK(
        is_nntile_device(self.device()),
        "nntile ",
        name,
        ": expected nntile");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile ",
        name,
        " supports float32 only");
    TORCH_CHECK(
        self.is_contiguous(),
        "nntile ",
        name,
        " requires contiguous input");
    if (out.has_value())
    {
        TORCH_CHECK(
            is_nntile_device(out->device()),
            "nntile ",
            name,
            ".out: expected nntile");
        TORCH_CHECK(
            out->sizes() == self.sizes(),
            "nntile ",
            name,
            ".out: shape mismatch");
        TORCH_CHECK(
            out->is_contiguous() &&
                out->scalar_type() == at::ScalarType::Float,
            "nntile ",
            name,
            ".out requires contiguous float32");
    }
}

} // namespace

at::Tensor cos_tensor(const at::Tensor &self)
{
    check_unary_fp32(self, "cos");
    at::Tensor out = at::empty_like(self);
    tensor_cos_fp32(self, out);
    return out;
}

at::Tensor &cos_out(const at::Tensor &self, at::Tensor &out)
{
    check_unary_fp32(self, "cos", out);
    tensor_cos_fp32(self, out);
    return out;
}

at::Tensor sin_tensor(const at::Tensor &self)
{
    check_unary_fp32(self, "sin");
    at::Tensor out = at::empty_like(self);
    tensor_sin_fp32(self, out);
    return out;
}

at::Tensor &sin_out(const at::Tensor &self, at::Tensor &out)
{
    check_unary_fp32(self, "sin", out);
    tensor_sin_fp32(self, out);
    return out;
}

at::Tensor neg_tensor(const at::Tensor &self)
{
    check_unary_fp32(self, "neg");
    at::Tensor out = at::empty_like(self);
    tensor_neg_fp32(self, out);
    return out;
}

at::Tensor &neg_out(const at::Tensor &self, at::Tensor &out)
{
    check_unary_fp32(self, "neg", out);
    tensor_neg_fp32(self, out);
    return out;
}

at::Tensor rsqrt_tensor(const at::Tensor &self)
{
    check_unary_fp32(self, "rsqrt");
    at::Tensor out = at::empty_like(self);
    tensor_rsqrt_fp32(self, out);
    return out;
}

at::Tensor &rsqrt_out(const at::Tensor &self, at::Tensor &out)
{
    check_unary_fp32(self, "rsqrt", out);
    tensor_rsqrt_fp32(self, out);
    return out;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("cos", TORCH_FN(torch_nntile::cos_tensor));
    m.impl("cos.out", TORCH_FN(torch_nntile::cos_out));
    m.impl("sin", TORCH_FN(torch_nntile::sin_tensor));
    m.impl("sin.out", TORCH_FN(torch_nntile::sin_out));
    m.impl("neg", TORCH_FN(torch_nntile::neg_tensor));
    m.impl("neg.out", TORCH_FN(torch_nntile::neg_out));
    m.impl("rsqrt", TORCH_FN(torch_nntile::rsqrt_tensor));
    m.impl("rsqrt.out", TORCH_FN(torch_nntile::rsqrt_out));
}
