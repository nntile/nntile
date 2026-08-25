/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_mul.cpp
 */

#include "nntile_executor.h"
#include "nntile_graph_recorder.h"
#include "nntile_graph_recorder_impl.h"
#include "nntile_no_implicit_copy.h"
#include "nntile_tensor_gc.h"

#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <torch/library.h>

namespace torch_nntile
{

namespace
{

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
    if (out.has_value())
    {
        TORCH_CHECK(
            out->sizes() == self.sizes(),
            "nntile mul.out: output shape mismatch");
    }
}

void run_mul_kernel(
    const at::Tensor &self,
    const at::Tensor &other,
    at::Tensor &out)
{
    tensor_mul_fp32(self, other, out);
}

} // namespace

at::Tensor mul_scalar(const at::Tensor &self, const at::Scalar &other)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "mul.Scalar", "self");
    if (self.scalar_type() == at::ScalarType::Long)
    {
        at::Tensor filled = empty_metadata_tensor(
            self.sizes(),
            at::kLong,
            self.device());
        tensor_fill_i64(filled, other.to<int64_t>());
        at::Tensor out = empty_metadata_tensor(
            self.sizes(),
            at::kLong,
            self.device());
        tensor_mul_i64(self, filled, out);
        return out;
    }
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile mul.Scalar supports float32 only");
    at::Tensor out = at::empty_like(self);
    tensor_mul_scalar_fp32(self, out, other.to<float>());
    return out;
}

at::Tensor &mul_scalar_out(
    const at::Tensor &self,
    const at::Scalar &other,
    at::Tensor &out)
{
    nntile::GraphFillScope record;
    TORCH_CHECK(
        is_nntile_device(self.device()) && is_nntile_device(out.device()),
        "nntile mul.Scalar_out expects nntile tensors");
    TORCH_CHECK(self.sizes() == out.sizes(), "nntile mul.Scalar_out shape");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float &&
            out.scalar_type() == at::ScalarType::Float,
        "nntile mul.Scalar_out supports float32 only");
    tensor_mul_scalar_fp32(self, out, other.to<float>());
    return out;
}

at::Tensor mul_fp32_bool(
    const at::Tensor &fp32,
    const at::Tensor &pred)
{
    require_nntile_operand(fp32, "mul.Tensor", "self");
    require_nntile_operand(pred, "mul.Tensor", "other");
    TORCH_CHECK(
        fp32.scalar_type() == at::kFloat && pred.scalar_type() == at::kBool,
        "nntile mul: expected float32 * bool");
    auto bcast = at::infer_size(fp32.sizes(), pred.sizes());
    at::Tensor zeros = at::zeros(
        bcast,
        fp32.options().memory_format(at::MemoryFormat::Contiguous));
    return at::where(pred, fp32, zeros);
}

at::Tensor mul_tensor(const at::Tensor &self, const at::Tensor &other)
{
    nntile::GraphFillScope record;
    // PyTorch may wrap Python floats as CPU 0-dim tensors for mul.Tensor.
    if (is_nntile_device(self.device()) && is_cpu_scalar_tensor(other))
    {
        return mul_scalar(self, other.item());
    }
    if (is_nntile_device(other.device()) && is_cpu_scalar_tensor(self))
    {
        return mul_scalar(other, self.item());
    }
    require_nntile_operand(self, "mul.Tensor", "self");
    require_nntile_operand(other, "mul.Tensor", "other");
    if (self.scalar_type() == at::kLong &&
        other.scalar_type() == at::kLong)
    {
        auto bcast = at::infer_size(self.sizes(), other.sizes());
        at::Tensor out = empty_metadata_tensor(
            bcast,
            at::kLong,
            self.device());
        tensor_mul_i64(self, other, out);
        return out;
    }
    if (self.scalar_type() == at::kFloat &&
        other.scalar_type() == at::kBool)
    {
        return mul_fp32_bool(self, other);
    }
    if (self.scalar_type() == at::kBool &&
        other.scalar_type() == at::kFloat)
    {
        return mul_fp32_bool(other, self);
    }
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float &&
            other.scalar_type() == at::ScalarType::Float,
        "nntile mul supports float32 only");
    std::vector<int64_t> out_sizes =
        at::infer_size(self.sizes(), other.sizes());
    at::Tensor out = at::empty(out_sizes, self.options());
    run_mul_kernel(self, other, out);
    return out;
}

at::Tensor &mul_out(
    const at::Tensor &self,
    const at::Tensor &other,
    at::Tensor &out)
{
    nntile::GraphFillScope record;
    if (is_nntile_device(self.device()) && is_cpu_scalar_tensor(other))
    {
        return mul_scalar_out(self, other.item(), out);
    }
    require_nntile_operand(self, "mul.out", "self");
    require_nntile_operand(other, "mul.out", "other");
    check_mul_inputs(self, other, out);
    run_mul_kernel(self, other, out);
    return out;
}

at::Tensor &mul_inplace_tensor(at::Tensor &self, const at::Tensor &other)
{
    nntile::GraphFillScope record;
    if (is_cpu_scalar_tensor(other) &&
        self.scalar_type() == at::ScalarType::Float)
    {
        at::Tensor tmp = mul_scalar(self, other.item());
        self.copy_(tmp);
        return self;
    }
    require_nntile_operand(self, "mul_.Tensor", "self");
    require_nntile_operand(other, "mul_.Tensor", "other");
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
