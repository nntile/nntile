/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_host_aten.cpp
 * Host-mediated aten ops for HF smokes (cast, triu, comparisons).
 *
 * Pattern: gather nntile → CPU aten → scatter back. Used for ops that
 * are awkward as StarPU fp32-only codelets (dtype cast, bool masks,
 * integer compare). CUDA StarPU workers still see the resulting nntile
 * buffers via normal ingress; the host step runs on the driver thread.
 */

#include "nntile_executor.h"
#include "nntile_graph_recorder.h"
#include "nntile_graph_recorder_impl.h"
#include "nntile_tensor_gc.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <ATen/ops/all.h>
#include <ATen/ops/div.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/eq.h>
#include <ATen/ops/gt.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/rsub.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/triu.h>
#include <ATen/ops/where.h>
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

bool is_cpu_scalar_tensor(const at::Tensor &t)
{
    return t.is_cpu() && t.numel() == 1;
}

at::Tensor gather_cpu(const at::Tensor &self)
{
    TORCH_CHECK(
        is_nntile_device(self.device()),
        "host_aten: expected nntile tensor");
    at::Tensor src = self.is_contiguous() ? self : self.contiguous();
    at::Tensor cpu = at::empty(
        src.sizes(),
        src.options().device(at::kCPU).memory_format(
            at::MemoryFormat::Contiguous));
    copy_nntile_tensor_to_cpu(src, cpu);
    return cpu;
}

at::Tensor scatter_nntile(
    const at::Tensor &cpu,
    c10::Device device)
{
    TORCH_CHECK(cpu.is_cpu(), "host_aten: expected CPU tensor");
    TORCH_CHECK(is_nntile_device(device), "host_aten: expected nntile device");
    at::Tensor contig = cpu.contiguous();
    at::Tensor out = empty_metadata_tensor(
        contig.sizes(),
        contig.scalar_type(),
        device);
    init_nntile_input_from_cpu(contig, out);
    return out;
}

void copy_cpu_into_nntile(const at::Tensor &cpu, at::Tensor &out)
{
    TORCH_CHECK(cpu.is_cpu(), "host_aten: expected CPU src");
    TORCH_CHECK(
        is_nntile_device(out.device()),
        "host_aten: expected nntile out");
    at::Tensor contig = cpu.contiguous();
    TORCH_CHECK(
        contig.sizes() == out.sizes(),
        "host_aten: out shape mismatch");
    TORCH_CHECK(
        contig.scalar_type() == out.scalar_type(),
        "host_aten: out dtype mismatch");
    if (!out.is_contiguous())
    {
        TORCH_CHECK(false, "host_aten: out must be contiguous");
    }
    init_nntile_input_from_cpu(contig, out);
}

} // namespace

at::Tensor to_copy(
    const at::Tensor &self,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    bool /*non_blocking*/,
    std::optional<at::MemoryFormat> /*memory_format*/)
{
    TORCH_CHECK(
        !pin_memory.has_value() || !*pin_memory,
        "_to_copy: pin_memory is CPU-only");
    if (layout.has_value())
    {
        TORCH_CHECK(
            *layout == at::kStrided,
            "_to_copy: only strided layout on nntile");
    }

    // PrivateUse1 may be selected for CPU→nntile moves (destination
    // device), so `self` is not always on nntile.
    at::Tensor cpu;
    if (is_nntile_device(self.device()))
    {
        cpu = gather_cpu(self);
    }
    else
    {
        TORCH_CHECK(
            self.is_cpu(),
            "_to_copy: expected CPU or nntile self");
        cpu = self.contiguous();
    }

    at::TensorOptions opts = cpu.options().device(at::kCPU);
    if (dtype.has_value())
    {
        opts = opts.dtype(*dtype);
    }
    at::Tensor converted =
        cpu.to(opts, /*non_blocking=*/false, /*copy=*/true);

    const c10::Device out_device =
        device.has_value() ? *device : self.device();
    if (out_device.is_cpu())
    {
        return converted;
    }
    TORCH_CHECK(
        is_nntile_device(out_device),
        "_to_copy: unsupported destination device");
    return scatter_nntile(converted, out_device);
}

at::Tensor triu(const at::Tensor &self, c10::SymInt diagonal)
{
    TORCH_CHECK(is_nntile_device(self.device()), "triu: expected nntile");
    at::Tensor cpu = gather_cpu(self);
    at::Tensor out_cpu =
        at::triu(cpu, diagonal.expect_int());
    return scatter_nntile(out_cpu, self.device());
}

at::Tensor &triu_out(
    const at::Tensor &self,
    c10::SymInt diagonal,
    at::Tensor &out)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) && is_nntile_device(out.device()),
        "triu.out: expected nntile");
    at::Tensor cpu = gather_cpu(self);
    at::Tensor tmp = at::triu(cpu, diagonal.expect_int());
    copy_cpu_into_nntile(tmp, out);
    return out;
}

at::Tensor eq_scalar(const at::Tensor &self, const at::Scalar &other)
{
    TORCH_CHECK(is_nntile_device(self.device()), "eq.Scalar: expected nntile");
    at::Tensor cpu = gather_cpu(self);
    at::Tensor out_cpu = at::eq(cpu, other);
    return scatter_nntile(out_cpu, self.device());
}

at::Tensor &eq_scalar_out(
    const at::Tensor &self,
    const at::Scalar &other,
    at::Tensor &out)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) && is_nntile_device(out.device()),
        "eq.Scalar_out: expected nntile");
    at::Tensor cpu = gather_cpu(self);
    at::Tensor tmp = at::eq(cpu, other);
    TORCH_CHECK(
        out.scalar_type() == tmp.scalar_type(),
        "eq.Scalar_out: dtype mismatch");
    TORCH_CHECK(out.sizes() == tmp.sizes(), "eq.Scalar_out: shape mismatch");
    copy_cpu_into_nntile(tmp, out);
    return out;
}

at::Tensor ne_scalar(const at::Tensor &self, const at::Scalar &other)
{
    TORCH_CHECK(is_nntile_device(self.device()), "ne.Scalar: expected nntile");
    at::Tensor cpu = gather_cpu(self);
    at::Tensor out_cpu = at::ne(cpu, other);
    return scatter_nntile(out_cpu, self.device());
}

at::Tensor &ne_scalar_out(
    const at::Tensor &self,
    const at::Scalar &other,
    at::Tensor &out)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) && is_nntile_device(out.device()),
        "ne.Scalar_out: expected nntile");
    at::Tensor cpu = gather_cpu(self);
    at::Tensor tmp = at::ne(cpu, other);
    TORCH_CHECK(
        out.scalar_type() == tmp.scalar_type(),
        "ne.Scalar_out: dtype mismatch");
    TORCH_CHECK(out.sizes() == tmp.sizes(), "ne.Scalar_out: shape mismatch");
    copy_cpu_into_nntile(tmp, out);
    return out;
}

at::Tensor all_tensor(const at::Tensor &self)
{
    TORCH_CHECK(is_nntile_device(self.device()), "all: expected nntile");
    // HF uses ``if torch.all(...)`` which needs a host scalar; keep the
    // reduction on CPU so ``.item()`` / ``__bool__`` does not require a
    // graph flush mid-forward.
    at::Tensor cpu = gather_cpu(self);
    return at::all(cpu);
}

at::Tensor &all_out(const at::Tensor &self, at::Tensor &out)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) && is_nntile_device(out.device()),
        "all.out: expected nntile");
    at::Tensor cpu = gather_cpu(self);
    at::Tensor tmp = at::all(cpu);
    TORCH_CHECK(
        out.scalar_type() == tmp.scalar_type(),
        "all.out: dtype mismatch");
    TORCH_CHECK(out.sizes() == tmp.sizes(), "all.out: shape mismatch");
    copy_cpu_into_nntile(tmp, out);
    return out;
}

at::Tensor gt_tensor(const at::Tensor &self, const at::Tensor &other)
{
    TORCH_CHECK(is_nntile_device(self.device()), "gt.Tensor: expected nntile");
    at::Tensor a = gather_cpu(self);
    at::Tensor b = is_nntile_device(other.device()) ? gather_cpu(other)
                                                    : other.cpu();
    return scatter_nntile(at::gt(a, b), self.device());
}

at::Tensor &gt_tensor_out(
    const at::Tensor &self,
    const at::Tensor &other,
    at::Tensor &out)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) && is_nntile_device(out.device()),
        "gt.Tensor_out: expected nntile");
    at::Tensor a = gather_cpu(self);
    at::Tensor b = is_nntile_device(other.device()) ? gather_cpu(other)
                                                    : other.cpu();
    at::Tensor tmp = at::gt(a, b);
    TORCH_CHECK(
        out.scalar_type() == tmp.scalar_type(),
        "gt.Tensor_out: dtype mismatch");
    TORCH_CHECK(out.sizes() == tmp.sizes(), "gt.Tensor_out: shape mismatch");
    copy_cpu_into_nntile(tmp, out);
    return out;
}

at::Tensor sub_scalar(
    const at::Tensor &self,
    const at::Scalar &other,
    const at::Scalar &alpha)
{
    TORCH_CHECK(is_nntile_device(self.device()), "sub.Scalar: expected nntile");
    at::Tensor cpu = gather_cpu(self);
    at::Tensor out_cpu = at::sub(cpu, other, alpha);
    return scatter_nntile(out_cpu, self.device());
}

at::Tensor &sub_scalar_out(
    const at::Tensor &self,
    const at::Scalar &other,
    const at::Scalar &alpha,
    at::Tensor &out)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) && is_nntile_device(out.device()),
        "sub.Scalar_out: expected nntile");
    at::Tensor cpu = gather_cpu(self);
    at::Tensor tmp = at::sub(cpu, other, alpha);
    TORCH_CHECK(out.sizes() == tmp.sizes(), "sub.Scalar_out: shape mismatch");
    TORCH_CHECK(
        out.scalar_type() == tmp.scalar_type(),
        "sub.Scalar_out: dtype mismatch");
    copy_cpu_into_nntile(tmp, out);
    return out;
}

at::Tensor rsub_scalar(
    const at::Tensor &self,
    const at::Scalar &other,
    const at::Scalar &alpha)
{
    TORCH_CHECK(
        is_nntile_device(self.device()),
        "rsub.Scalar: expected nntile");
    at::Tensor cpu = gather_cpu(self);
    at::Tensor out_cpu = at::rsub(cpu, other, alpha);
    return scatter_nntile(out_cpu, self.device());
}

at::Tensor &rsub_scalar_out(
    const at::Tensor &self,
    const at::Scalar &other,
    const at::Scalar &alpha,
    at::Tensor &out)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) && is_nntile_device(out.device()),
        "rsub.Scalar_out: expected nntile");
    at::Tensor cpu = gather_cpu(self);
    at::Tensor tmp = at::rsub(cpu, other, alpha);
    TORCH_CHECK(
        out.sizes() == tmp.sizes(),
        "rsub.Scalar_out: shape mismatch");
    TORCH_CHECK(
        out.scalar_type() == tmp.scalar_type(),
        "rsub.Scalar_out: dtype mismatch");
    copy_cpu_into_nntile(tmp, out);
    return out;
}

at::Tensor pow_tensor_scalar(
    const at::Tensor &self,
    const at::Scalar &exponent)
{
    TORCH_CHECK(is_nntile_device(self.device()), "pow: expected nntile");
    return scatter_nntile(
        at::pow(gather_cpu(self), exponent),
        self.device());
}

at::Tensor &pow_tensor_scalar_out(
    const at::Tensor &self,
    const at::Scalar &exponent,
    at::Tensor &out)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) && is_nntile_device(out.device()),
        "pow.out: expected nntile");
    at::Tensor tmp = at::pow(gather_cpu(self), exponent);
    at::Tensor scattered = scatter_nntile(tmp, self.device());
    out.copy_(scattered);
    return out;
}

at::Tensor mean_dim(
    const at::Tensor &self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> /*dtype*/)
{
    TORCH_CHECK(is_nntile_device(self.device()), "mean: expected nntile");
    at::Tensor cpu = gather_cpu(self);
    at::Tensor out_cpu = dim.has_value()
        ? at::mean(cpu, *dim, keepdim)
        : at::mean(cpu);
    return scatter_nntile(out_cpu, self.device());
}

at::Tensor &mean_out(
    const at::Tensor &self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> /*dtype*/,
    at::Tensor &out)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) && is_nntile_device(out.device()),
        "mean.out: expected nntile");
    at::Tensor cpu = gather_cpu(self);
    at::Tensor tmp = dim.has_value() ? at::mean(cpu, *dim, keepdim)
                                     : at::mean(cpu);
    at::Tensor scattered = scatter_nntile(tmp, self.device());
    out.copy_(scattered);
    return out;
}

at::Tensor div_scalar(
    const at::Tensor &self,
    const at::Scalar &other)
{
    TORCH_CHECK(is_nntile_device(self.device()), "div.Scalar: expected nntile");
    // Prefer mul by reciprocal so StarPU MulScalar handles fp32 compute.
    const double v = other.toDouble();
    TORCH_CHECK(v != 0.0, "div.Scalar: division by zero");
    at::Tensor out = at::empty_like(self);
    tensor_mul_scalar_fp32(self, out, static_cast<float>(1.0 / v));
    return out;
}

at::Tensor &div_scalar_out(
    const at::Tensor &self,
    const at::Scalar &other,
    at::Tensor &out)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) && is_nntile_device(out.device()),
        "div.Scalar_out: expected nntile");
    const double v = other.toDouble();
    TORCH_CHECK(v != 0.0, "div.Scalar_out: division by zero");
    tensor_mul_scalar_fp32(self, out, static_cast<float>(1.0 / v));
    return out;
}

at::Tensor div_tensor(const at::Tensor &self, const at::Tensor &other)
{
    if (is_cpu_scalar_tensor(other))
    {
        return div_scalar(self, other.item());
    }
    TORCH_CHECK(
        is_nntile_device(self.device()) && is_nntile_device(other.device()),
        "div.Tensor: expected nntile");
    at::Tensor a = gather_cpu(self);
    at::Tensor b = gather_cpu(other);
    return scatter_nntile(at::div(a, b), self.device());
}

at::Tensor &div_out(
    const at::Tensor &self,
    const at::Tensor &other,
    at::Tensor &out)
{
    if (is_cpu_scalar_tensor(other))
    {
        return div_scalar_out(self, other.item(), out);
    }
    at::Tensor tmp = div_tensor(self, other);
    out.copy_(tmp);
    return out;
}

at::Tensor where_self(
    const at::Tensor &condition,
    const at::Tensor &self,
    const at::Tensor &other)
{
    at::Tensor cond_cpu = is_nntile_device(condition.device())
        ? gather_cpu(condition)
        : condition.cpu();
    at::Tensor self_cpu = is_nntile_device(self.device()) ? gather_cpu(self)
                                                          : self.cpu();
    at::Tensor other_cpu = is_nntile_device(other.device())
        ? gather_cpu(other)
        : other.cpu();
    at::Tensor out_cpu = at::where(cond_cpu, self_cpu, other_cpu);
    const c10::Device out_dev = is_nntile_device(self.device())
        ? self.device()
        : (is_nntile_device(other.device()) ? other.device()
                                            : condition.device());
    TORCH_CHECK(
        is_nntile_device(out_dev),
        "where.self: expected nntile destination");
    return scatter_nntile(out_cpu, out_dev);
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("_to_copy", TORCH_FN(torch_nntile::to_copy));
    m.impl("triu", TORCH_FN(torch_nntile::triu));
    m.impl("triu.out", TORCH_FN(torch_nntile::triu_out));
    m.impl("eq.Scalar", TORCH_FN(torch_nntile::eq_scalar));
    m.impl("eq.Scalar_out", TORCH_FN(torch_nntile::eq_scalar_out));
    m.impl("ne.Scalar", TORCH_FN(torch_nntile::ne_scalar));
    m.impl("ne.Scalar_out", TORCH_FN(torch_nntile::ne_scalar_out));
    m.impl("all", TORCH_FN(torch_nntile::all_tensor));
    m.impl("all.all_out", TORCH_FN(torch_nntile::all_out));
    m.impl("gt.Tensor", TORCH_FN(torch_nntile::gt_tensor));
    m.impl("gt.Tensor_out", TORCH_FN(torch_nntile::gt_tensor_out));
    m.impl("sub.Scalar", TORCH_FN(torch_nntile::sub_scalar));
    m.impl("sub.Scalar_out", TORCH_FN(torch_nntile::sub_scalar_out));
    m.impl("rsub.Scalar", TORCH_FN(torch_nntile::rsub_scalar));
    m.impl("rsub.Scalar_out", TORCH_FN(torch_nntile::rsub_scalar_out));
    m.impl("pow.Tensor_Scalar", TORCH_FN(torch_nntile::pow_tensor_scalar));
    m.impl(
        "pow.Tensor_Scalar_out",
        TORCH_FN(torch_nntile::pow_tensor_scalar_out));
    m.impl("mean.dim", TORCH_FN(torch_nntile::mean_dim));
    m.impl("mean.out", TORCH_FN(torch_nntile::mean_out));
    m.impl("div.Scalar", TORCH_FN(torch_nntile::div_scalar));
    m.impl("div.Scalar_out", TORCH_FN(torch_nntile::div_scalar_out));
    m.impl("div.Tensor", TORCH_FN(torch_nntile::div_tensor));
    m.impl("div.out", TORCH_FN(torch_nntile::div_out));
    m.impl("where.self", TORCH_FN(torch_nntile::where_self));
}
