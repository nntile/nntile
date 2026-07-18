/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_sub.cpp
 * Out-of-place aten::sub for device=nntile (torch-native StarPU path).
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

at::Tensor scatter_nntile(
    const at::Tensor &cpu,
    c10::Device device)
{
    TORCH_CHECK(cpu.is_cpu(), "nntile sub: expected CPU tensor");
    at::Tensor contig = cpu.contiguous();
    at::Tensor out = empty_metadata_tensor(
        contig.sizes(),
        contig.scalar_type(),
        device);
    init_nntile_input_from_cpu(contig, out);
    return out;
}

at::Tensor sub_host(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha)
{
    at::Tensor a = gather_nntile_view_to_cpu(self);
    at::Tensor b = is_nntile_device(other.device())
        ? gather_nntile_view_to_cpu(other)
        : other.cpu();
    return scatter_nntile(at::sub(a, b, alpha), self.device());
}

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
    TORCH_CHECK(
        self.is_contiguous() && other.is_contiguous(),
        "nntile sub requires contiguous tensors");
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

at::Tensor sub_tensor(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha)
{
    // T5 relative bias: long broadcast ``memory - context``.
    if (!is_nntile_device(other.device()) ||
        self.scalar_type() != other.scalar_type() ||
        self.scalar_type() != at::ScalarType::Float ||
        !self.sizes().equals(other.sizes()))
    {
        return sub_host(self, other, alpha);
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
    if (!is_nntile_device(other.device()) ||
        self.scalar_type() != other.scalar_type() ||
        self.scalar_type() != at::ScalarType::Float ||
        !self.sizes().equals(other.sizes()))
    {
        at::Tensor tmp = sub_host(self, other, alpha);
        out.copy_(tmp);
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
        out.scalar_type() == at::ScalarType::Float &&
            out.is_contiguous(),
        "nntile sub.out requires contiguous float32 output");
    run_torch_sub(self, other, alpha, out);
    return out;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("sub.Tensor", TORCH_FN(torch_nntile::sub_tensor));
    m.impl("sub.out", TORCH_FN(torch_nntile::sub_out));
}
