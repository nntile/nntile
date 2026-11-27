/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_add.cpp
 * PrivateUse1 ``aten::add`` (torch autograd InputBuffer / AccumulateGrad
 * and user ``torch.add``). The classic ``nntile::tensor::add`` kernels
 * handle float32; int64 is not supported.
 */

#include "nntile_graph_recorder.h"
#include "nntile_graph_recorder_impl.h"
#include "nntile_no_implicit_copy.h"
#include "nntile_tensor_gc.h"

#include <nntile/defs.h>

#include "nntile_executor_classic.h"

#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

namespace torch_nntile
{

namespace
{

void check_add_dtypes(
    const at::Tensor &self,
    const at::Tensor &other)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) &&
            is_nntile_device(other.device()),
        "nntile add expects both operands on device nntile");
    TORCH_CHECK(
        self.scalar_type() == other.scalar_type(),
        "nntile add: dtype mismatch");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile add supports float32 only");
}

void require_classic_add_fp32(
    const at::Tensor &self,
    const at::Tensor &other,
    char const *op)
{
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float &&
            other.scalar_type() == at::ScalarType::Float,
        "nntile ",
        op,
        ": classic add supports float32 only");
    TORCH_CHECK(
        self.is_contiguous() && other.is_contiguous(),
        "nntile ",
        op,
        ": classic add requires contiguous tensors");
    TORCH_CHECK(
        self.sizes().equals(other.sizes()),
        "nntile ",
        op,
        ": classic add requires equal shapes (no broadcast)");
}

void classic_add_out(
    const at::Tensor &self,
    const at::Tensor &other,
    float alpha,
    at::Tensor &out)
{
    require_classic_add_fp32(self, other, "add");
    TORCH_CHECK(
        out.sizes().equals(self.sizes()) &&
            out.scalar_type() == at::ScalarType::Float,
        "nntile add: output shape/dtype mismatch");
    classic_tensor_add_fp32(1.0f, self, alpha, other, out);
}

void classic_add_inplace(
    at::Tensor &self,
    const at::Tensor &other,
    float alpha)
{
    require_classic_add_fp32(self, other, "add_");
    classic_tensor_add_inplace_fp32(alpha, other, 1.0f, self);
}

void run_add(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha,
    at::Tensor &out)
{
    classic_add_out(self, other, alpha.to<float>(), out);
}

} // namespace

at::Tensor add_scalar(
    const at::Tensor &self,
    const at::Scalar &other,
    const at::Scalar &alpha)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "add.Scalar", "self");
    if (self.scalar_type() == at::ScalarType::Long)
    {
        TORCH_CHECK(
            false,
            "nntile add.Scalar int64 is not supported by the classic "
            "nntile kernel stack");
    }
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile add.Scalar supports float32 only");
    TORCH_CHECK(
        self.is_contiguous(),
        "nntile add.Scalar: classic add requires a contiguous tensor");
    at::Tensor filled = at::empty_like(self);
    at::Tensor out = at::empty_like(self);
    classic_tensor_fill_fp32(
        filled,
        other.to<float>() * alpha.to<float>());
    classic_tensor_add_fp32(1.0f, self, 1.0f, filled, out);
    return out;
}

at::Tensor &add_scalar_out(
    const at::Tensor &self,
    const at::Scalar &other,
    const at::Scalar &alpha,
    at::Tensor &out)
{
    nntile::GraphFillScope record;
    TORCH_CHECK(
        is_nntile_device(self.device()) && is_nntile_device(out.device()),
        "nntile add.Scalar_out expects nntile");
    at::Tensor tmp = add_scalar(self, other, alpha);
    out.copy_(tmp);
    return out;
}

at::Tensor add_tensor(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha)
{
    nntile::GraphFillScope record;
    if (is_nntile_device(self.device()) && is_cpu_scalar_tensor(other))
    {
        return add_scalar(self, other.item(), alpha);
    }
    if (is_cpu_scalar_tensor(self) && is_nntile_device(other.device()))
    {
        return add_scalar(other, self.item(), alpha);
    }
    require_nntile_operand(self, "add.Tensor", "self");
    require_nntile_operand(other, "add.Tensor", "other");
    if (self.scalar_type() == at::kLong &&
        other.scalar_type() == at::kLong)
    {
        TORCH_CHECK(
            false,
            "nntile add.Tensor int64 is not supported by the classic "
            "nntile kernel stack");
    }
    check_add_dtypes(self, other);
    std::vector<int64_t> out_sizes =
        at::infer_size(self.sizes(), other.sizes());
    at::Tensor out = at::empty(
        out_sizes,
        self.options().memory_format(at::MemoryFormat::Contiguous));
    run_add(self, other, alpha, out);
    return out;
}

at::Tensor &add_out(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha,
    at::Tensor &out)
{
    nntile::GraphFillScope record;
    if (is_nntile_device(self.device()) && is_cpu_scalar_tensor(other))
    {
        return add_scalar_out(self, other.item(), alpha, out);
    }
    require_nntile_operand(self, "add.out", "self");
    require_nntile_operand(other, "add.out", "other");
    check_add_dtypes(self, other);
    TORCH_CHECK(
        is_nntile_device(out.device()),
        "nntile add.out expects output on device nntile");
    TORCH_CHECK(
        out.sizes().equals(
            at::infer_size(self.sizes(), other.sizes())),
        "nntile add.out: output shape mismatch");
    TORCH_CHECK(
        out.scalar_type() == at::ScalarType::Float,
        "nntile add.out expects float32 output");
    run_add(self, other, alpha, out);
    return out;
}

at::Tensor &add__tensor(
    at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha)
{
    nntile::GraphFillScope record;
    if (is_cpu_scalar_tensor(other))
    {
        at::Tensor tmp = add_scalar(self, other.item(), alpha);
        self.copy_(tmp);
        return self;
    }
    require_nntile_operand(self, "add_.Tensor", "self");
    require_nntile_operand(other, "add_.Tensor", "other");
    if (self.scalar_type() == at::kLong &&
        other.scalar_type() == at::kLong)
    {
        TORCH_CHECK(
            false,
            "nntile add_.Tensor int64 is not supported by the classic "
            "nntile kernel stack");
    }
    check_add_dtypes(self, other);
    TORCH_CHECK(
        self.sizes().equals(
            at::infer_size(self.sizes(), other.sizes())),
        "nntile add_.Tensor: other must broadcast to self");
    classic_add_inplace(self, other, alpha.to<float>());
    return self;
}

at::Tensor &add__scalar(
    at::Tensor &self,
    const at::Scalar &other,
    const at::Scalar &alpha)
{
    nntile::GraphFillScope record;
    at::Tensor tmp = add_scalar(self, other, alpha);
    self.copy_(tmp);
    return self;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("add.Tensor", TORCH_FN(torch_nntile::add_tensor));
    m.impl("add.out", TORCH_FN(torch_nntile::add_out));
    m.impl("add_.Tensor", TORCH_FN(torch_nntile::add__tensor));
    m.impl("add_.Scalar", TORCH_FN(torch_nntile::add__scalar));
    m.impl("add.Scalar", TORCH_FN(torch_nntile::add_scalar));
    m.impl("add.Scalar_out", TORCH_FN(torch_nntile::add_scalar_out));
}
