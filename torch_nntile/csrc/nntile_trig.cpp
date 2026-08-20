/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_trig.cpp
 * Torch-native cos / sin / neg / rsqrt / exp (StarPU unary family).
 */

#include "nntile_executor.h"
#include "nntile_tensor_gc.h"

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

at::Tensor neg_tensor(const at::Tensor &self)
{
    TORCH_CHECK(
        is_nntile_device(self.device()),
        "nntile neg: expected nntile");
    if (self.scalar_type() == at::kLong)
    {
        at::Tensor out = empty_metadata_tensor(
            self.sizes(),
            at::kLong,
            self.device());
        tensor_neg_i64(self, out);
        return out;
    }
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile neg supports float32 and int64 only "
        "(implicit host copy disabled)");
    check_unary_fp32(self.is_contiguous() ? self : self.contiguous(), "neg");
    at::Tensor inp = self.is_contiguous() ? self : self.contiguous();
    at::Tensor out = at::empty_like(inp);
    tensor_neg_fp32(inp, out);
    return out;
}

at::Tensor &neg_out(const at::Tensor &self, at::Tensor &out)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) &&
            is_nntile_device(out.device()),
        "nntile neg.out: expected nntile");
    if (self.scalar_type() == at::kLong)
    {
        TORCH_CHECK(
            out.scalar_type() == at::kLong &&
                out.sizes() == self.sizes(),
            "nntile neg.out: int64 shape mismatch");
        tensor_neg_i64(self, out);
        return out;
    }
    at::Tensor inp = self.is_contiguous() ? self : self.contiguous();
    check_unary_fp32(inp, "neg", out);
    tensor_neg_fp32(inp, out);
    return out;
}

at::Tensor cos_tensor(const at::Tensor &self)
{
    at::Tensor inp = self.is_contiguous() ? self : self.contiguous();
    check_unary_fp32(inp, "cos");
    at::Tensor out = at::empty_like(inp);
    tensor_cos_fp32(inp, out);
    return out;
}

at::Tensor &cos_out(const at::Tensor &self, at::Tensor &out)
{
    at::Tensor inp = self.is_contiguous() ? self : self.contiguous();
    check_unary_fp32(inp, "cos", out);
    tensor_cos_fp32(inp, out);
    return out;
}

at::Tensor sin_tensor(const at::Tensor &self)
{
    at::Tensor inp = self.is_contiguous() ? self : self.contiguous();
    check_unary_fp32(inp, "sin");
    at::Tensor out = at::empty_like(inp);
    tensor_sin_fp32(inp, out);
    return out;
}

at::Tensor &sin_out(const at::Tensor &self, at::Tensor &out)
{
    at::Tensor inp = self.is_contiguous() ? self : self.contiguous();
    check_unary_fp32(inp, "sin", out);
    tensor_sin_fp32(inp, out);
    return out;
}

at::Tensor rsqrt_tensor(const at::Tensor &self)
{
    at::Tensor inp = self.is_contiguous() ? self : self.contiguous();
    check_unary_fp32(inp, "rsqrt");
    at::Tensor out = at::empty_like(inp);
    tensor_rsqrt_fp32(inp, out);
    return out;
}

at::Tensor &rsqrt_out(const at::Tensor &self, at::Tensor &out)
{
    at::Tensor inp = self.is_contiguous() ? self : self.contiguous();
    check_unary_fp32(inp, "rsqrt", out);
    tensor_rsqrt_fp32(inp, out);
    return out;
}

at::Tensor exp_tensor(const at::Tensor &self)
{
    at::Tensor inp = self.is_contiguous() ? self : self.contiguous();
    check_unary_fp32(inp, "exp");
    at::Tensor out = at::empty_like(inp);
    tensor_exp_fp32(inp, out);
    return out;
}

at::Tensor &exp_out(const at::Tensor &self, at::Tensor &out)
{
    at::Tensor inp = self.is_contiguous() ? self : self.contiguous();
    check_unary_fp32(inp, "exp", out);
    tensor_exp_fp32(inp, out);
    return out;
}

at::Tensor log_tensor(const at::Tensor &self)
{
    at::Tensor inp = self.is_contiguous() ? self : self.contiguous();
    check_unary_fp32(inp, "log");
    at::Tensor out = at::empty_like(inp);
    tensor_log_fp32(inp, out);
    return out;
}

at::Tensor &log_out(const at::Tensor &self, at::Tensor &out)
{
    at::Tensor inp = self.is_contiguous() ? self : self.contiguous();
    check_unary_fp32(inp, "log", out);
    tensor_log_fp32(inp, out);
    return out;
}

} // namespace torch_nntile

// Match device=cuda: PrivateUse1 (device) forward only. Autograd uses the
// generic VariableType formula ``-0.5 * grad * result.pow(3)`` (see
// derivatives.yaml); keep ``pow.Tensor_Scalar`` on StarPU for exp 2/3.
// ``exp`` likewise: VariableType ``grad * result`` (no AutogradPrivateUse1).
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
    m.impl("exp", TORCH_FN(torch_nntile::exp_tensor));
    m.impl("exp.out", TORCH_FN(torch_nntile::exp_out));
    m.impl("log", TORCH_FN(torch_nntile::log_tensor));
    m.impl("log.out", TORCH_FN(torch_nntile::log_out));
}
