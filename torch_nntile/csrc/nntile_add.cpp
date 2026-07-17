/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_add.cpp
 * Out-of-place aten::add for device=nntile (torch-native StarPU path).
 */

#include "nntile_executor.h"
#include "nntile_graph_recorder.h"
#include "nntile_graph_recorder_impl.h"
#include "nntile_tensor_gc.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
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
        "nntile add: expected nntile tensor");
    return gather_nntile_view_to_cpu(self);
}

at::Tensor scatter_nntile(
    const at::Tensor &cpu,
    c10::Device device)
{
    TORCH_CHECK(cpu.is_cpu(), "nntile add: expected CPU tensor");
    at::Tensor contig = cpu.contiguous();
    at::Tensor out = empty_metadata_tensor(
        contig.sizes(),
        contig.scalar_type(),
        device);
    init_nntile_input_from_cpu(contig, out);
    return out;
}

void check_add_inputs(
    const at::Tensor &self,
    const at::Tensor &other)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) &&
            is_nntile_device(other.device()),
        "nntile add expects both operands on device nntile");
    TORCH_CHECK(
        self.sizes().equals(other.sizes()),
        "nntile torch_add: same-shape tensors only "
        "(broadcast disabled under NNTILE_TORCH_NATIVE_OPS)");
    TORCH_CHECK(
        self.scalar_type() == other.scalar_type(),
        "nntile add: dtype mismatch");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nntile torch_add supports float32 only");
    TORCH_CHECK(
        self.is_contiguous() && other.is_contiguous(),
        "nntile add requires contiguous tensors");
}

void run_torch_add(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha,
    at::Tensor &out)
{
    // tensor_add_fp32(alpha_x, x, beta_y, y, out) with alpha_x=1,
    // beta_y = torch alpha → out = x + alpha * y.
    tensor_add_fp32(
        1.0f,
        self,
        alpha.to<float>(),
        other,
        out);
}

at::Tensor add_host(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha)
{
    at::Tensor a = gather_cpu(self);
    at::Tensor b = is_nntile_device(other.device()) ? gather_cpu(other)
                                                    : other.cpu();
    return scatter_nntile(at::add(a, b, alpha), self.device());
}

} // namespace

at::Tensor add_scalar(
    const at::Tensor &self,
    const at::Scalar &other,
    const at::Scalar &alpha)
{
    TORCH_CHECK(
        is_nntile_device(self.device()),
        "nntile add.Scalar expects nntile self");
    at::Tensor cpu = gather_cpu(self);
    return scatter_nntile(at::add(cpu, other, alpha), self.device());
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
    if (!is_nntile_device(other.device()) ||
        self.scalar_type() != other.scalar_type() ||
        self.scalar_type() != at::ScalarType::Float ||
        !self.sizes().equals(other.sizes()))
    {
        return add_host(self, other, alpha);
    }
    check_add_inputs(self, other);
    at::Tensor out = at::empty(
        self.sizes(),
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
    if (!is_nntile_device(other.device()) ||
        self.scalar_type() != other.scalar_type() ||
        self.scalar_type() != at::ScalarType::Float ||
        !self.sizes().equals(other.sizes()))
    {
        at::Tensor tmp = add_host(self, other, alpha);
        out.copy_(tmp);
        return out;
    }
    check_add_inputs(self, other);
    TORCH_CHECK(
        is_nntile_device(out.device()),
        "nntile add.out expects output on device nntile");
    TORCH_CHECK(
        out.sizes().equals(self.sizes()),
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
    if (!is_nntile_device(other.device()) ||
        self.scalar_type() != other.scalar_type() ||
        self.scalar_type() != at::ScalarType::Float ||
        !self.sizes().equals(other.sizes()))
    {
        at::Tensor tmp = add_host(self, other, alpha);
        self.copy_(tmp);
        return self;
    }
    check_add_inputs(self, other);
    // SSA: record out-of-place add and rebind ``self`` to the result node.
    tensor_add_inplace_fp32(
        alpha.to<float>(),
        other,
        1.0f,
        self);
    return self;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("add.Tensor", TORCH_FN(torch_nntile::add_tensor));
    m.impl("add.out", TORCH_FN(torch_nntile::add_out));
    m.impl("add_.Tensor", TORCH_FN(torch_nntile::add__tensor));
    m.impl("add.Scalar", TORCH_FN(torch_nntile::add_scalar));
    m.impl("add.Scalar_out", TORCH_FN(torch_nntile::add_scalar_out));
}
