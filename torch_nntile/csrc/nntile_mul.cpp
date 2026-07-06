/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_mul.cpp
 */

#include "nntile_executor.h"
#include "nntile_graph_recorder_impl.h"

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
    pin_graph_op_inputs({self, other});
    pin_graph_op_output(out, true);
    tensor_mul_fp32(
        self.data_ptr<float>(),
        other.data_ptr<float>(),
        out.data_ptr<float>(),
        self.sizes());
}

void run_mul_inplace_kernel(at::Tensor &self, const at::Tensor &other)
{
    pin_graph_op_inputs({self, other});
    pin_graph_op_output(self, true);
    tensor_mul_inplace_fp32(
        other.data_ptr<float>(),
        self.data_ptr<float>(),
        self.sizes());
}

} // namespace

at::Tensor mul_tensor(const at::Tensor &self, const at::Tensor &other)
{
    check_mul_inputs(self, other);
    at::Tensor out = at::empty_like(self);
    run_mul_kernel(self, other, out);
    return out;
}

at::Tensor &mul_out(
    const at::Tensor &self,
    const at::Tensor &other,
    at::Tensor &out)
{
    check_mul_inputs(self, other, out);
    run_mul_kernel(self, other, out);
    return out;
}

at::Tensor &mul_inplace_tensor(at::Tensor &self, const at::Tensor &other)
{
    check_mul_inputs(self, other);
    run_mul_inplace_kernel(self, other);
    return self;
}

at::Tensor mul_scalar(const at::Tensor &self, const at::Scalar &other)
{
    TORCH_CHECK(
        is_nntile_device(self.device()),
        "nntile mul.Scalar expects tensor on device nntile");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile mul.Scalar supports float32 only");
    TORCH_CHECK(self.is_contiguous(), "nntile mul.Scalar requires contiguous");
    at::Tensor out = at::empty_like(self);
    tensor_mul_scalar_fp32(
        self.data_ptr<float>(),
        out.data_ptr<float>(),
        self.sizes(),
        other.to<float>());
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
    TORCH_CHECK(
        self.is_contiguous() && out.is_contiguous(),
        "nntile mul.Scalar_out requires contiguous tensors");
    tensor_mul_scalar_fp32(
        self.data_ptr<float>(),
        out.data_ptr<float>(),
        self.sizes(),
        other.to<float>());
    return out;
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
