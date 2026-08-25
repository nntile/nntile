/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_sub.cpp
 * Out-of-place aten::sub for device=nntile (torch-native StarPU path).
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

at::Tensor sub_scalar(
    const at::Tensor &self,
    const at::Scalar &other,
    const at::Scalar &alpha)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "sub.Scalar", "self");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile sub.Scalar supports float32 only");
    at::Tensor filled = at::empty_like(self);
    tensor_fill_fp32(filled, other.to<float>());
    at::Tensor out = at::empty_like(self);
    tensor_sub_fp32(self, filled, alpha.to<float>(), out);
    return out;
}

at::Tensor &sub_scalar_out(
    const at::Tensor &self,
    const at::Scalar &other,
    const at::Scalar &alpha,
    at::Tensor &out)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "sub.Scalar_out", "self");
    require_nntile_operand(out, "sub.Scalar_out", "out");
    at::Tensor tmp = sub_scalar(self, other, alpha);
    out.copy_(tmp);
    return out;
}

at::Tensor rsub_scalar(
    const at::Tensor &self,
    const at::Scalar &other,
    const at::Scalar &alpha)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "rsub.Scalar", "self");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile rsub.Scalar supports float32 only");
    at::Tensor filled = at::empty_like(self);
    tensor_fill_fp32(filled, other.to<float>());
    at::Tensor out = at::empty_like(self);
    tensor_sub_fp32(filled, self, alpha.to<float>(), out);
    return out;
}

at::Tensor &rsub_scalar_out(
    const at::Tensor &self,
    const at::Scalar &other,
    const at::Scalar &alpha,
    at::Tensor &out)
{
    nntile::GraphFillScope record;
    require_nntile_operand(self, "rsub.Scalar_out", "self");
    require_nntile_operand(out, "rsub.Scalar_out", "out");
    at::Tensor tmp = rsub_scalar(self, other, alpha);
    out.copy_(tmp);
    return out;
}

at::Tensor sub_tensor(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha)
{
    nntile::GraphFillScope record;
    if (is_nntile_device(self.device()) && is_cpu_scalar_tensor(other))
    {
        return sub_scalar(self, other.item(), alpha);
    }
    if (is_cpu_scalar_tensor(self) && is_nntile_device(other.device()))
    {
        return rsub_scalar(other, self.item(), alpha);
    }
    require_nntile_operand(self, "sub.Tensor", "self");
    require_nntile_operand(other, "sub.Tensor", "other");
    if (self.scalar_type() == at::kLong &&
        other.scalar_type() == at::kLong)
    {
        TORCH_CHECK(
            alpha.to<double>() == 1.0,
            "nntile sub.Tensor int64: alpha must be 1");
        auto bcast = at::infer_size(self.sizes(), other.sizes());
        at::Tensor out = empty_metadata_tensor(
            bcast,
            at::kLong,
            self.device());
        tensor_sub_i64(self, other, out);
        return out;
    }
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
    nntile::GraphFillScope record;
    if (is_nntile_device(self.device()) && is_cpu_scalar_tensor(other))
    {
        return sub_scalar_out(self, other.item(), alpha, out);
    }
    if (is_cpu_scalar_tensor(self) && is_nntile_device(other.device()))
    {
        return rsub_scalar_out(other, self.item(), alpha, out);
    }
    require_nntile_operand(self, "sub.out", "self");
    require_nntile_operand(other, "sub.out", "other");
    if (self.scalar_type() == at::kLong &&
        other.scalar_type() == at::kLong &&
        out.scalar_type() == at::kLong)
    {
        TORCH_CHECK(
            alpha.to<double>() == 1.0,
            "nntile sub.out int64: alpha must be 1");
        tensor_sub_i64(self, other, out);
        return out;
    }
    check_sub_inputs(self, other);
    TORCH_CHECK(
        is_nntile_device(out.device()),
        "nntile sub.out expects output on device nntile");
    TORCH_CHECK(
        out.sizes().equals(self.sizes()),
        "nntile sub.out: output shape mismatch");
    TORCH_CHECK(
        out.scalar_type() == at::ScalarType::Float,
        "nntile sub.out expects float32 output");
    run_torch_sub(self, other, alpha, out);
    return out;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("sub.Tensor", TORCH_FN(torch_nntile::sub_tensor));
    m.impl("sub.out", TORCH_FN(torch_nntile::sub_out));
    m.impl("sub.Scalar", TORCH_FN(torch_nntile::sub_scalar));
    m.impl("sub.Scalar_out", TORCH_FN(torch_nntile::sub_scalar_out));
    m.impl("rsub.Scalar", TORCH_FN(torch_nntile::rsub_scalar));
    m.impl("rsub.Scalar_out", TORCH_FN(torch_nntile::rsub_scalar_out));
}
