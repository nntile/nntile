/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_mul.cpp
 */

#include "nntile_executor.h"
#include "nntile_graph_recorder.h"
#include "nntile_graph_recorder_impl.h"
#include "nntile_tensor_gc.h"

#include <ATen/ExpandUtils.h>
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

at::Tensor gather_cpu(const at::Tensor &self)
{
    TORCH_CHECK(
        is_nntile_device(self.device()),
        "nntile mul: expected nntile tensor");
    return gather_nntile_view_to_cpu(self);
}

at::Tensor scatter_nntile(
    const at::Tensor &cpu,
    c10::Device device)
{
    TORCH_CHECK(cpu.is_cpu(), "nntile mul: expected CPU tensor");
    at::Tensor contig = cpu.contiguous();
    at::Tensor out = empty_metadata_tensor(
        contig.sizes(),
        contig.scalar_type(),
        device);
    init_nntile_input_from_cpu(contig, out);
    return out;
}

bool is_cpu_scalar_tensor(const at::Tensor &t)
{
    return t.is_cpu() && t.numel() == 1;
}

void check_mul_inputs(
    const at::Tensor &self,
    const at::Tensor &other,
    const std::optional<at::Tensor> &out = std::nullopt)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) &&
            is_nntile_device(other.device()),
        "nntile mul expects both operands on device nntile");
    if (out.has_value())
    {
        TORCH_CHECK(
            is_nntile_device(out->device()),
            "nntile mul.out expects output on device nntile");
    }
    TORCH_CHECK(self.sizes() == other.sizes(), "nntile mul: shape mismatch");
    TORCH_CHECK(
        self.scalar_type() == other.scalar_type(),
        "nntile mul: dtype mismatch");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile mul supports float32 only in phase 2");
    TORCH_CHECK(
        self.is_contiguous() && other.is_contiguous(),
        "nntile mul requires contiguous tensors");
    if (out.has_value())
    {
        TORCH_CHECK(
            out->sizes() == self.sizes(),
            "nntile mul.out: output shape mismatch");
        TORCH_CHECK(
            out->is_contiguous(),
            "nntile mul.out requires contiguous output");
    }
}

void run_mul_kernel(
    const at::Tensor &self,
    const at::Tensor &other,
    at::Tensor &out)
{
    tensor_mul_fp32(self, other, out);
}

void run_mul_inplace_kernel(at::Tensor &self, const at::Tensor &other)
{
    tensor_mul_inplace_fp32(other, self);
}

at::Tensor mul_host(const at::Tensor &self, const at::Tensor &other)
{
    at::Tensor a = gather_cpu(self);
    at::Tensor b = is_nntile_device(other.device()) ? gather_cpu(other)
                                                    : other.cpu();
    return scatter_nntile(at::mul(a, b), self.device());
}

void mul_inplace_host(at::Tensor &self, const at::Tensor &other)
{
    at::Tensor result = mul_host(self, other);
    // SSA-style rebind: nntile←nntile copy attaches the result TensorRef.
    self.copy_(result);
}

} // namespace

at::Tensor mul_scalar(const at::Tensor &self, const at::Scalar &other)
{
    TORCH_CHECK(
        is_nntile_device(self.device()),
        "nntile mul.Scalar expects tensor on device nntile");
    if (self.scalar_type() != at::ScalarType::Float)
    {
        return scatter_nntile(
            at::mul(gather_cpu(self), other),
            self.device());
    }
    at::Tensor inp = self.is_contiguous() ? self : self.contiguous();
    at::Tensor out = at::empty_like(inp);
    tensor_mul_scalar_fp32(inp, out, other.to<float>());
    return out;
}

at::Tensor &mul_scalar_out(
    const at::Tensor &self,
    const at::Scalar &other,
    at::Tensor &out)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) && is_nntile_device(out.device()),
        "nntile mul.Scalar_out expects nntile tensors");
    TORCH_CHECK(self.sizes() == out.sizes(), "nntile mul.Scalar_out shape");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float &&
            out.scalar_type() == at::ScalarType::Float,
        "nntile mul.Scalar_out supports float32 only");
    at::Tensor inp = self.is_contiguous() ? self : self.contiguous();
    TORCH_CHECK(
        out.is_contiguous(),
        "nntile mul.Scalar_out requires contiguous output");
    tensor_mul_scalar_fp32(inp, out, other.to<float>());
    return out;
}

at::Tensor mul_tensor(const at::Tensor &self, const at::Tensor &other)
{
    // PyTorch may wrap Python floats as CPU 0-dim tensors for mul.Tensor.
    if (is_nntile_device(self.device()) && is_cpu_scalar_tensor(other))
    {
        return mul_scalar(self, other.item());
    }
    if (is_nntile_device(other.device()) && is_cpu_scalar_tensor(self))
    {
        return mul_scalar(other, self.item());
    }
    if (!is_nntile_device(other.device()) ||
        self.scalar_type() != other.scalar_type() ||
        self.scalar_type() != at::ScalarType::Float)
    {
        return mul_host(self, other);
    }
    std::vector<int64_t> out_sizes =
        at::infer_size(self.sizes(), other.sizes());
    at::Tensor a = self.sizes().equals(out_sizes)
        ? (self.is_contiguous() ? self : self.contiguous())
        : self.expand(out_sizes).contiguous();
    at::Tensor b = other.sizes().equals(out_sizes)
        ? (other.is_contiguous() ? other : other.contiguous())
        : other.expand(out_sizes).contiguous();
    check_mul_inputs(a, b);
    at::Tensor out = at::empty(
        out_sizes,
        a.options().memory_format(at::MemoryFormat::Contiguous));
    run_mul_kernel(a, b, out);
    return out;
}

at::Tensor &mul_out(
    const at::Tensor &self,
    const at::Tensor &other,
    at::Tensor &out)
{
    if (is_nntile_device(self.device()) && is_cpu_scalar_tensor(other))
    {
        return mul_scalar_out(self, other.item(), out);
    }
    if (!is_nntile_device(other.device()) ||
        self.scalar_type() != other.scalar_type() ||
        self.scalar_type() != at::ScalarType::Float)
    {
        at::Tensor tmp = mul_host(self, other);
        out.copy_(tmp);
        return out;
    }
    check_mul_inputs(self, other, out);
    run_mul_kernel(self, other, out);
    return out;
}

at::Tensor &mul_inplace_tensor(at::Tensor &self, const at::Tensor &other)
{
    if (is_cpu_scalar_tensor(other) &&
        self.scalar_type() == at::ScalarType::Float)
    {
        at::Tensor tmp = mul_scalar(self, other.item());
        self.copy_(tmp);
        return self;
    }
    if (!is_nntile_device(other.device()) ||
        self.scalar_type() != other.scalar_type() ||
        self.scalar_type() != at::ScalarType::Float)
    {
        mul_inplace_host(self, other);
        return self;
    }
    at::Tensor tmp = mul_tensor(self, other);
    self.copy_(tmp);
    return self;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("mul.Tensor", TORCH_FN(torch_nntile::mul_tensor));
    m.impl("mul.out", TORCH_FN(torch_nntile::mul_out));
    m.impl("mul_.Tensor", TORCH_FN(torch_nntile::mul_inplace_tensor));
    m.impl("mul.Scalar", TORCH_FN(torch_nntile::mul_scalar));
    m.impl("mul.Scalar_out", TORCH_FN(torch_nntile::mul_scalar_out));
}
