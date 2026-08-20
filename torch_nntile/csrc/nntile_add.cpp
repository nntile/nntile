/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_add.cpp
 * Out-of-place aten::add for device=nntile (torch-native StarPU path).
 */

#include "nntile_executor.h"
#include "nntile_graph_recorder.h"
#include "nntile_graph_recorder_impl.h"
#include "nntile_no_implicit_copy.h"
#include "nntile_tensor_gc.h"

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
        "nntile torch_add supports float32 only");
}

void run_torch_add(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha,
    at::Tensor &out)
{
    // Keep view strides (including broadcast 0-strides). at::add_out
    // in the StarPU codelet broadcasts; do not densify via contiguous().
    tensor_add_fp32(
        1.0f,
        self,
        alpha.to<float>(),
        other,
        out);
}

} // namespace

at::Tensor add_scalar(
    const at::Tensor &self,
    const at::Scalar &other,
    const at::Scalar &alpha)
{
    require_nntile_operand(self, "add.Scalar", "self");
    if (self.scalar_type() == at::ScalarType::Long)
    {
        TORCH_CHECK(
            alpha.to<double>() == 1.0,
            "nntile add.Scalar int64: alpha must be 1");
        at::Tensor filled = empty_metadata_tensor(
            self.sizes(),
            at::kLong,
            self.device());
        tensor_fill_i64(filled, other.to<int64_t>());
        at::Tensor out = empty_metadata_tensor(
            self.sizes(),
            at::kLong,
            self.device());
        tensor_add_i64(self, filled, out);
        return out;
    }
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile add.Scalar supports float32 only");
    at::Tensor inp = self.is_contiguous() ? self : self.contiguous();
    at::Tensor filled = at::empty_like(inp);
    tensor_fill_fp32(
        filled,
        other.to<float>() * alpha.to<float>());
    at::Tensor out = at::empty_like(inp);
    tensor_add_fp32(1.0f, inp, 1.0f, filled, out);
    return out;
}

at::Tensor &add_scalar_out(
    const at::Tensor &self,
    const at::Scalar &other,
    const at::Scalar &alpha,
    at::Tensor &out)
{
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
            alpha.to<double>() == 1.0,
            "nntile add.Tensor int64: alpha must be 1");
        auto bcast = at::infer_size(self.sizes(), other.sizes());
        at::Tensor out = empty_metadata_tensor(
            bcast,
            at::kLong,
            self.device());
        tensor_add_i64(self, other, out);
        return out;
    }
    check_add_dtypes(self, other);
    std::vector<int64_t> out_sizes =
        at::infer_size(self.sizes(), other.sizes());
    at::Tensor out = at::empty(
        out_sizes,
        self.options().memory_format(at::MemoryFormat::Contiguous));
    run_torch_add(self, other, alpha, out);
    return out;
}

at::Tensor &add_out(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha,
    at::Tensor &out)
{
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
        out.scalar_type() == at::ScalarType::Float &&
            out.is_contiguous(),
        "nntile add.out requires contiguous float32 output");
    run_torch_add(self, other, alpha, out);
    return out;
}

at::Tensor &add__tensor(
    at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha)
{
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
            alpha.to<double>() == 1.0,
            "nntile add_.Tensor int64: alpha must be 1");
        at::Tensor tmp = empty_metadata_tensor(
            self.sizes(),
            at::kLong,
            self.device());
        tensor_add_i64(self, other, tmp);
        self.copy_(tmp);
        return self;
    }
    check_add_dtypes(self, other);
    TORCH_CHECK(
        self.sizes().equals(
            at::infer_size(self.sizes(), other.sizes())),
        "nntile add_.Tensor: other must broadcast to self");
    // SSA: record out-of-place add and rebind ``self`` to the result node.
    tensor_add_inplace_fp32(
        alpha.to<float>(),
        other,
        1.0f,
        self);
    return self;
}

at::Tensor &add__scalar(
    at::Tensor &self,
    const at::Scalar &other,
    const at::Scalar &alpha)
{
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
