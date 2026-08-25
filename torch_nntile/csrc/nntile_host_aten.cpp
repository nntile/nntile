/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_host_aten.cpp
 * Explicit device copies (``aten::_to_copy``) plus graph-native leftovers
 * that used to silently gather/scatter (``where``, ``full_like``, ``pow``
 * 2/3, ``div.Scalar``).
 *
 * Compute kernels must not move nntile payloads to CPU. The user moves
 * tensors with ``.to("nntile")`` / ``.to("cpu")``. Unregistered ops error
 * unless ``cpu_fallback=True`` (explicit opt-in).
 */

#include "nntile_executor.h"
#include "nntile_graph_recorder.h"
#include "nntile_graph_recorder_impl.h"
#include "nntile_no_implicit_copy.h"
#include "nntile_tensor_gc.h"

#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <ATen/ops/full.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

namespace torch_nntile
{

namespace
{

at::Tensor scatter_nntile(
    const at::Tensor &cpu,
    c10::Device device)
{
    TORCH_CHECK(cpu.is_cpu(), "_to_copy: expected CPU tensor");
    TORCH_CHECK(
        is_nntile_device(device),
        "_to_copy: expected nntile device");
    at::Tensor contig = cpu.contiguous();
    at::Tensor out = empty_metadata_tensor(
        contig.sizes(),
        contig.scalar_type(),
        device);
    init_nntile_input_from_cpu(contig, out);
    return out;
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
    nntile::GraphFillScope record;
    TORCH_CHECK(
        !pin_memory.has_value() || !*pin_memory,
        "_to_copy: pin_memory is CPU-only");
    if (layout.has_value())
    {
        TORCH_CHECK(
            *layout == at::kStrided,
            "_to_copy: only strided layout on nntile");
    }

    // Same-device dtype change: StarPU from_blob + copy_, no gather.
    const c10::Device out_device =
        device.has_value() ? *device : self.device();
    if (is_nntile_device(self.device()) &&
        is_nntile_device(out_device))
    {
        const at::ScalarType out_dtype =
            dtype.value_or(self.scalar_type());
        if (out_dtype == self.scalar_type())
        {
            return self;
        }
        at::Tensor out = empty_metadata_tensor(
            self.sizes(),
            out_dtype,
            out_device);
        tensor_cast(self, out);
        return out;
    }

    // Explicit ``.to()``: nntile→CPU gathers; CPU→nntile scatters.
    at::Tensor cpu;
    if (is_nntile_device(self.device()))
    {
        cpu = gather_nntile_view_to_cpu(self);
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

    if (out_device.is_cpu())
    {
        return converted;
    }
    TORCH_CHECK(
        is_nntile_device(out_device),
        "_to_copy: unsupported destination device");
    return scatter_nntile(converted, out_device);
}

at::Tensor full_like_tensor(
    const at::Tensor &self,
    const at::Scalar &fill_value,
    std::optional<at::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> /*memory_format*/)
{
    nntile::GraphFillScope record;
    TORCH_CHECK(
        !pin_memory.has_value() || !*pin_memory,
        "full_like: pin_memory is CPU-only");
    if (layout.has_value())
    {
        TORCH_CHECK(
            *layout == at::kStrided,
            "full_like: only strided layout on nntile");
    }
    const c10::ScalarType out_dtype =
        dtype.value_or(self.scalar_type());
    const c10::Device out_dev = device.value_or(self.device());
    if (out_dev.is_cpu())
    {
        return at::full(
            self.sizes(),
            fill_value,
            at::TensorOptions()
                .dtype(out_dtype)
                .device(at::kCPU)
                .layout(at::kStrided));
    }
    require_nntile_operand(self, "full_like", "self");
    TORCH_CHECK(
        is_nntile_device(out_dev),
        "full_like: unsupported destination device");
    at::Tensor out = empty_metadata_tensor(
        self.sizes(),
        out_dtype,
        out_dev);
    out.fill_(fill_value);
    return out;
}

at::Tensor zeros_like_tensor(
    const at::Tensor &self,
    std::optional<at::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> memory_format)
{
    nntile::GraphFillScope record;
    return full_like_tensor(
        self,
        /*fill_value=*/0,
        dtype,
        layout,
        device,
        pin_memory,
        memory_format);
}

at::Tensor pow_tensor_scalar(
    const at::Tensor &self,
    const at::Scalar &exponent)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "pow", "self");
    TORCH_CHECK(
        self.scalar_type() == at::kFloat,
        "torch_nntile pow: float32 only");
    at::Tensor out = at::empty_like(self);
    tensor_pow_scalar_fp32(self, out, exponent.to<float>());
    return out;
}

at::Tensor &pow_tensor_scalar_out(
    const at::Tensor &self,
    const at::Scalar &exponent,
    at::Tensor &out)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "pow.out", "self");
    require_nntile_operand(out, "pow.out", "out");
    at::Tensor tmp = pow_tensor_scalar(self, exponent);
    out.copy_(tmp);
    return out;
}

at::Tensor div_scalar(
    const at::Tensor &self,
    const at::Scalar &other)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "div.Scalar", "self");
    const double v = other.toDouble();
    TORCH_CHECK(v != 0.0, "div.Scalar: division by zero");
    TORCH_CHECK(
        self.scalar_type() == at::kFloat,
        "div.Scalar: float32 only");
    at::Tensor out = at::empty_like(self);
    tensor_mul_scalar_fp32(self, out, static_cast<float>(1.0 / v));
    return out;
}

at::Tensor &div_scalar_out(
    const at::Tensor &self,
    const at::Scalar &other,
    at::Tensor &out)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "div.Scalar_out", "self");
    require_nntile_operand(out, "div.Scalar_out", "out");
    const double v = other.toDouble();
    TORCH_CHECK(v != 0.0, "div.Scalar_out: division by zero");
    TORCH_CHECK(
        self.scalar_type() == at::kFloat &&
            out.scalar_type() == at::kFloat,
        "div.Scalar_out: float32 only");
    tensor_mul_scalar_fp32(self, out, static_cast<float>(1.0 / v));
    return out;
}

at::Tensor div_tensor(const at::Tensor &self, const at::Tensor &other)
{
    nntile::GraphFillScope record;
    if (is_cpu_scalar_tensor(other))
    {
        return div_scalar(self, other.item());
    }
    require_nntile_operand(self, "div.Tensor", "self");
    require_nntile_operand(other, "div.Tensor", "other");
    TORCH_CHECK(
        self.scalar_type() == at::kFloat &&
            other.scalar_type() == at::kFloat,
        "torch_nntile div.Tensor: float32 only");
    auto bcast = at::infer_size(self.sizes(), other.sizes());
    at::Tensor out = empty_metadata_tensor(
        bcast,
        at::kFloat,
        self.device());
    tensor_div_fp32(self, other, out);
    return out;
}

at::Tensor &div_out(
    const at::Tensor &self,
    const at::Tensor &other,
    at::Tensor &out)
{
    nntile::GraphFillScope record;
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
    nntile::GraphFillScope record;
    require_nntile_operand(condition, "where.self", "condition");
    require_nntile_operand(self, "where.self", "self");
    at::Tensor other_op = other;
    if (is_cpu_scalar_tensor(other) &&
        other.scalar_type() == at::kFloat &&
        self.scalar_type() == at::kFloat)
    {
        // Python/ATen may wrap a float as a 0-dim CPU tensor. Promote
        // onto nntile with FILL (no gather of ``self``).
        other_op = empty_metadata_tensor(
            {},
            at::kFloat,
            self.device());
        other_op.fill_(other.item());
    }
    else if (
        is_cpu_scalar_tensor(other) &&
        other.scalar_type() == at::kLong &&
        self.scalar_type() == at::kLong)
    {
        other_op = empty_metadata_tensor(
            {},
            at::kLong,
            self.device());
        tensor_fill_i64(other_op, other.item().to<int64_t>());
    }
    else
    {
        require_nntile_operand(other, "where.self", "other");
    }
    TORCH_CHECK(
        condition.scalar_type() == at::kBool,
        "where.self: bool condition");
    if (self.scalar_type() == at::kLong &&
        other_op.scalar_type() == at::kLong)
    {
        auto bcast = at::infer_size(self.sizes(), other_op.sizes());
        bcast = at::infer_size(bcast, condition.sizes());
        at::Tensor out = empty_metadata_tensor(
            bcast,
            at::kLong,
            self.device());
        tensor_where_i64(condition, self, other_op, out);
        return out;
    }
    TORCH_CHECK(
        self.scalar_type() == at::kFloat &&
            other_op.scalar_type() == at::kFloat,
        "where.self: bool condition and float32 or int64 self/other");
    auto bcast = at::infer_size(self.sizes(), other_op.sizes());
    bcast = at::infer_size(bcast, condition.sizes());
    at::Tensor out = empty_metadata_tensor(
        bcast,
        at::kFloat,
        self.device());
    tensor_where_fp32(condition, self, other_op, out);
    return out;
}

at::Tensor triu_tensor(const at::Tensor &self, int64_t diagonal)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "triu", "self");
    TORCH_CHECK(
        self.scalar_type() == at::kFloat,
        "torch_nntile triu: float32 only");
    TORCH_CHECK(self.dim() >= 2, "torch_nntile triu: expected ndim >= 2");
    at::Tensor out = at::empty_like(self);
    tensor_triu_fp32(self, out, diagonal);
    return out;
}

at::Tensor &triu_out(
    const at::Tensor &self,
    int64_t diagonal,
    at::Tensor &out)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "triu.out", "self");
    require_nntile_operand(out, "triu.out", "out");
    TORCH_CHECK(
        self.scalar_type() == at::kFloat &&
            out.scalar_type() == at::kFloat,
        "torch_nntile triu.out: float32 only");
    TORCH_CHECK(self.sizes() == out.sizes(), "triu.out: shape mismatch");
    tensor_triu_fp32(self, out, diagonal);
    return out;
}

at::Tensor tril_tensor(const at::Tensor &self, int64_t diagonal)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "tril", "self");
    TORCH_CHECK(
        self.scalar_type() == at::kBool,
        "torch_nntile tril: bool only");
    TORCH_CHECK(self.dim() >= 2, "torch_nntile tril: expected ndim >= 2");
    at::Tensor out = at::empty_like(self);
    tensor_tril_bool(self, out, diagonal);
    return out;
}

at::Tensor &tril_out(
    const at::Tensor &self,
    int64_t diagonal,
    at::Tensor &out)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "tril.out", "self");
    require_nntile_operand(out, "tril.out", "out");
    TORCH_CHECK(
        self.scalar_type() == at::kBool &&
            out.scalar_type() == at::kBool,
        "torch_nntile tril.out: bool only");
    TORCH_CHECK(self.sizes() == out.sizes(), "tril.out: shape mismatch");
    tensor_tril_bool(self, out, diagonal);
    return out;
}

at::Tensor gt_tensor(const at::Tensor &self, const at::Tensor &other)
{
    nntile::GraphFillScope record;
    at::Tensor rhs = other;
    if (is_nntile_device(self.device()) && is_cpu_scalar_tensor(other))
    {
        require_nntile_operand(self, "gt.Tensor", "self");
        TORCH_CHECK(
            self.scalar_type() == at::kLong,
            "torch_nntile gt.Tensor: int64 only");
        rhs = empty_metadata_tensor(
            {},
            at::kLong,
            self.device());
        tensor_fill_i64(rhs, other.item().to<int64_t>());
    }
    require_nntile_operand(self, "gt.Tensor", "self");
    require_nntile_operand(rhs, "gt.Tensor", "other");
    TORCH_CHECK(
        self.scalar_type() == at::kLong &&
            rhs.scalar_type() == at::kLong,
        "torch_nntile gt.Tensor: int64 only");
    auto bcast = at::infer_size(self.sizes(), rhs.sizes());
    at::Tensor out = empty_metadata_tensor(
        bcast,
        at::kBool,
        self.device());
    tensor_gt_i64(self, rhs, out);
    return out;
}

at::Tensor &gt_out(
    const at::Tensor &self,
    const at::Tensor &other,
    at::Tensor &out)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "gt.out", "self");
    require_nntile_operand(other, "gt.out", "other");
    require_nntile_operand(out, "gt.out", "out");
    TORCH_CHECK(
        self.scalar_type() == at::kLong &&
            other.scalar_type() == at::kLong &&
            out.scalar_type() == at::kBool,
        "torch_nntile gt.out: int64 inputs, bool out");
    tensor_gt_i64(self, other, out);
    return out;
}

at::Tensor filled_i64_scalar(
    const at::Tensor &like,
    int64_t value)
{
    at::Tensor scalar = empty_metadata_tensor(
        {},
        at::kLong,
        like.device());
    tensor_fill_i64(scalar, value);
    return scalar;
}

at::Tensor gt_scalar(const at::Tensor &self, const at::Scalar &other)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "gt.Scalar", "self");
    TORCH_CHECK(
        self.scalar_type() == at::kLong,
        "torch_nntile gt.Scalar: int64 only");
    return gt_tensor(self, filled_i64_scalar(self, other.to<int64_t>()));
}

at::Tensor lt_tensor(const at::Tensor &self, const at::Tensor &other)
{
    nntile::GraphFillScope record;
    at::Tensor rhs = other;
    if (is_nntile_device(self.device()) && is_cpu_scalar_tensor(other))
    {
        require_nntile_operand(self, "lt.Tensor", "self");
        TORCH_CHECK(
            self.scalar_type() == at::kLong,
            "torch_nntile lt.Tensor: int64 only");
        rhs = empty_metadata_tensor(
            {},
            at::kLong,
            self.device());
        tensor_fill_i64(rhs, other.item().to<int64_t>());
    }
    require_nntile_operand(self, "lt.Tensor", "self");
    require_nntile_operand(rhs, "lt.Tensor", "other");
    TORCH_CHECK(
        self.scalar_type() == at::kLong &&
            rhs.scalar_type() == at::kLong,
        "torch_nntile lt.Tensor: int64 only");
    auto bcast = at::infer_size(self.sizes(), rhs.sizes());
    at::Tensor out = empty_metadata_tensor(
        bcast,
        at::kBool,
        self.device());
    tensor_lt_i64(self, rhs, out);
    return out;
}

at::Tensor &lt_out(
    const at::Tensor &self,
    const at::Tensor &other,
    at::Tensor &out)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "lt.out", "self");
    require_nntile_operand(other, "lt.out", "other");
    require_nntile_operand(out, "lt.out", "out");
    tensor_lt_i64(self, other, out);
    return out;
}

at::Tensor lt_scalar(const at::Tensor &self, const at::Scalar &other)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "lt.Scalar", "self");
    TORCH_CHECK(
        self.scalar_type() == at::kLong,
        "torch_nntile lt.Scalar: int64 only");
    return lt_tensor(self, filled_i64_scalar(self, other.to<int64_t>()));
}

at::Tensor abs_tensor(const at::Tensor &self)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "abs", "self");
    TORCH_CHECK(
        self.scalar_type() == at::kLong,
        "torch_nntile abs: int64 only");
    at::Tensor out = empty_metadata_tensor(
        self.sizes(),
        at::kLong,
        self.device());
    tensor_abs_i64(self, out);
    return out;
}

at::Tensor &abs_out(const at::Tensor &self, at::Tensor &out)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "abs.out", "self");
    require_nntile_operand(out, "abs.out", "out");
    tensor_abs_i64(self, out);
    return out;
}

at::Tensor minimum_tensor(
    const at::Tensor &self,
    const at::Tensor &other)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "minimum.Tensor", "self");
    require_nntile_operand(other, "minimum.Tensor", "other");
    TORCH_CHECK(
        self.scalar_type() == at::kLong &&
            other.scalar_type() == at::kLong,
        "torch_nntile minimum: int64 only");
    auto bcast = at::infer_size(self.sizes(), other.sizes());
    at::Tensor out = empty_metadata_tensor(
        bcast,
        at::kLong,
        self.device());
    tensor_minimum_i64(self, other, out);
    return out;
}

at::Tensor &minimum_out(
    const at::Tensor &self,
    const at::Tensor &other,
    at::Tensor &out)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "minimum.out", "self");
    require_nntile_operand(other, "minimum.out", "other");
    require_nntile_operand(out, "minimum.out", "out");
    tensor_minimum_i64(self, other, out);
    return out;
}

at::Tensor filled_fp32_scalar(
    const at::Tensor &like,
    const at::Scalar &value)
{
    at::Tensor scalar = empty_metadata_tensor(
        {},
        at::kFloat,
        like.device());
    scalar.fill_(value);
    return scalar;
}

at::Tensor eq_tensor(
    const at::Tensor &self,
    const at::Tensor &other)
{
    nntile::GraphFillScope record;
    at::Tensor rhs = other;
    if (is_nntile_device(self.device()) && is_cpu_scalar_tensor(other))
    {
        require_nntile_operand(self, "eq.Tensor", "self");
        TORCH_CHECK(
            self.scalar_type() == at::kFloat,
            "torch_nntile eq.Tensor: float32 only");
        rhs = filled_fp32_scalar(self, other.item());
    }
    require_nntile_operand(self, "eq.Tensor", "self");
    require_nntile_operand(rhs, "eq.Tensor", "other");
    TORCH_CHECK(
        self.scalar_type() == at::kFloat &&
            rhs.scalar_type() == at::kFloat,
        "torch_nntile eq.Tensor: float32 only");
    auto bcast = at::infer_size(self.sizes(), rhs.sizes());
    at::Tensor out = empty_metadata_tensor(
        bcast,
        at::kBool,
        self.device());
    tensor_eq_fp32(self, rhs, out);
    return out;
}

at::Tensor &eq_out(
    const at::Tensor &self,
    const at::Tensor &other,
    at::Tensor &out)
{
    nntile::GraphFillScope record;
    if (is_nntile_device(self.device()) && is_cpu_scalar_tensor(other))
    {
        at::Tensor rhs = filled_fp32_scalar(self, other.item());
        require_nntile_operand(out, "eq.out", "out");
        tensor_eq_fp32(self, rhs, out);
        return out;
    }
    require_nntile_operand(self, "eq.out", "self");
    require_nntile_operand(other, "eq.out", "other");
    require_nntile_operand(out, "eq.out", "out");
    tensor_eq_fp32(self, other, out);
    return out;
}

at::Tensor eq_scalar(
    const at::Tensor &self,
    const at::Scalar &other)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "eq.Scalar", "self");
    TORCH_CHECK(
        self.scalar_type() == at::kFloat,
        "torch_nntile eq.Scalar: float32 only");
    return eq_tensor(self, filled_fp32_scalar(self, other));
}

at::Tensor &eq_scalar_out(
    const at::Tensor &self,
    const at::Scalar &other,
    at::Tensor &out)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "eq.Scalar_out", "self");
    require_nntile_operand(out, "eq.Scalar_out", "out");
    TORCH_CHECK(
        self.scalar_type() == at::kFloat,
        "torch_nntile eq.Scalar_out: float32 only");
    at::Tensor rhs = filled_fp32_scalar(self, other);
    tensor_eq_fp32(self, rhs, out);
    return out;
}

at::Tensor masked_fill_scalar(
    const at::Tensor &self,
    const at::Tensor &mask,
    const at::Scalar &value)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "masked_fill.Scalar", "self");
    require_nntile_operand(mask, "masked_fill.Scalar", "mask");
    TORCH_CHECK(
        self.scalar_type() == at::kFloat,
        "torch_nntile masked_fill: float32 self");
    TORCH_CHECK(
        mask.scalar_type() == at::kBool,
        "torch_nntile masked_fill: bool mask");
    at::Tensor filled = empty_metadata_tensor(
        self.sizes(),
        at::kFloat,
        self.device());
    filled.fill_(value);
    return where_self(mask, filled, self);
}

at::Tensor &masked_fill__scalar(
    at::Tensor &self,
    const at::Tensor &mask,
    const at::Scalar &value)
{
    nntile::GraphFillScope record;
    at::Tensor tmp = masked_fill_scalar(self, mask, value);
    self.copy_(tmp);
    return self;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("_to_copy", TORCH_FN(torch_nntile::to_copy));
    m.impl("full_like", TORCH_FN(torch_nntile::full_like_tensor));
    m.impl("zeros_like", TORCH_FN(torch_nntile::zeros_like_tensor));
    m.impl("pow.Tensor_Scalar", TORCH_FN(torch_nntile::pow_tensor_scalar));
    m.impl(
        "pow.Tensor_Scalar_out",
        TORCH_FN(torch_nntile::pow_tensor_scalar_out));
    m.impl("div.Scalar", TORCH_FN(torch_nntile::div_scalar));
    m.impl("div.Scalar_out", TORCH_FN(torch_nntile::div_scalar_out));
    m.impl("div.Tensor", TORCH_FN(torch_nntile::div_tensor));
    m.impl("div.out", TORCH_FN(torch_nntile::div_out));
    m.impl("where.self", TORCH_FN(torch_nntile::where_self));
    m.impl("triu", TORCH_FN(torch_nntile::triu_tensor));
    m.impl("triu.out", TORCH_FN(torch_nntile::triu_out));
    m.impl("tril", TORCH_FN(torch_nntile::tril_tensor));
    m.impl("tril.out", TORCH_FN(torch_nntile::tril_out));
    m.impl("gt.Tensor", TORCH_FN(torch_nntile::gt_tensor));
    m.impl("gt.out", TORCH_FN(torch_nntile::gt_out));
    m.impl("gt.Scalar", TORCH_FN(torch_nntile::gt_scalar));
    m.impl("lt.Tensor", TORCH_FN(torch_nntile::lt_tensor));
    m.impl("lt.out", TORCH_FN(torch_nntile::lt_out));
    m.impl("lt.Scalar", TORCH_FN(torch_nntile::lt_scalar));
    m.impl("abs", TORCH_FN(torch_nntile::abs_tensor));
    m.impl("abs.out", TORCH_FN(torch_nntile::abs_out));
    m.impl("minimum.Tensor", TORCH_FN(torch_nntile::minimum_tensor));
    m.impl("minimum.out", TORCH_FN(torch_nntile::minimum_out));
    m.impl("min.other", TORCH_FN(torch_nntile::minimum_tensor));
    m.impl("eq.Tensor", TORCH_FN(torch_nntile::eq_tensor));
    m.impl("eq.out", TORCH_FN(torch_nntile::eq_out));
    m.impl("eq.Scalar", TORCH_FN(torch_nntile::eq_scalar));
    m.impl("eq.Scalar_out", TORCH_FN(torch_nntile::eq_scalar_out));
    m.impl(
        "masked_fill.Scalar",
        TORCH_FN(torch_nntile::masked_fill_scalar));
    m.impl(
        "masked_fill_.Scalar",
        TORCH_FN(torch_nntile::masked_fill__scalar));
}
