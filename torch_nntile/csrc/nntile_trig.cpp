/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_trig.cpp
 * Torch-native cos / sin / neg / rsqrt (StarPU unary family).
 */

#include "nntile_executor.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <torch/csrc/autograd/custom_function.h>
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
    check_unary_fp32(self.is_contiguous() ? self : self.contiguous(), "neg");
    at::Tensor inp = self.is_contiguous() ? self : self.contiguous();
    at::Tensor out = at::empty_like(inp);
    tensor_neg_fp32(inp, out);
    return out;
}

at::Tensor &neg_out(const at::Tensor &self, at::Tensor &out)
{
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

//! ``d(rsqrt(x))/dx = -0.5 * rsqrt(x)^3`` — keep Llama RMSNorm on StarPU.
class RsqrtFn : public torch::autograd::Function<RsqrtFn>
{
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        at::Tensor self)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        at::Tensor out = rsqrt_tensor(self);
        ctx->save_for_backward({out});
        return out;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        const at::Tensor out = ctx->get_saved_variables()[0];
        const at::Tensor &grad = grad_outputs[0];
        at::Tensor out2 = at::empty_like(out);
        tensor_mul_fp32(out, out, out2);
        at::Tensor out3 = at::empty_like(out);
        tensor_mul_fp32(out2, out, out3);
        at::Tensor scaled = at::empty_like(out);
        tensor_mul_scalar_fp32(out3, scaled, -0.5f);
        at::Tensor dx = at::empty_like(out);
        tensor_mul_fp32(grad, scaled, dx);
        return {dx};
    }
};

at::Tensor rsqrt_autograd(const at::Tensor &self)
{
    TORCH_CHECK(
        is_nntile_device(self.device()),
        "nntile rsqrt: expected nntile");
    return RsqrtFn::apply(self);
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

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m)
{
    m.impl("rsqrt", TORCH_FN(torch_nntile::rsqrt_autograd));
}
