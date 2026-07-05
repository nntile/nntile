/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_arange.cpp
 */

#include "nntile_allocator.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <cstring>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

void copy_cpu_tensor_to_nntile(const at::Tensor &cpu, at::Tensor &nntile_out)
{
    TORCH_CHECK(cpu.is_cpu(), "expected CPU staging tensor");
    TORCH_CHECK(
        is_nntile_device(nntile_out.device()),
        "expected nntile output tensor");
    TORCH_CHECK(
        cpu.sizes() == nntile_out.sizes(),
        "nntile arange: output shape mismatch");
    TORCH_CHECK(
        cpu.scalar_type() == nntile_out.scalar_type(),
        "nntile arange: dtype mismatch");
    TORCH_CHECK(
        cpu.is_contiguous() && nntile_out.is_contiguous(),
        "nntile arange requires contiguous tensors");
    const std::size_t nbytes = static_cast<std::size_t>(cpu.nbytes());
    if (nbytes == 0)
    {
        return;
    }
    std::memcpy(
        nntile_out.storage().data_ptr().get(),
        cpu.storage().data_ptr().get(),
        nbytes);
}

at::Tensor &arange_start_step_out_impl(
    const at::Scalar &start,
    const at::Scalar &end,
    const at::Scalar &step,
    at::Tensor &out)
{
    TORCH_CHECK(
        is_nntile_device(out.device()),
        "nntile arange.out expects output on device nntile");
    const at::Tensor cpu_out =
        at::arange(start, end, step, out.options().device(at::kCPU));
    if (out.sizes() != cpu_out.sizes())
    {
        out.resize_(cpu_out.sizes());
    }
    TORCH_CHECK(out.is_contiguous(), "nntile arange.out requires contiguous out");
    copy_cpu_tensor_to_nntile(cpu_out, out);
    return out;
}

} // namespace

at::Tensor &arange_start_step_out(
    const at::Scalar &start,
    const at::Scalar &end,
    const at::Scalar &step,
    at::Tensor &out)
{
    return arange_start_step_out_impl(start, end, step, out);
}

at::Tensor arange_start_step(
    const at::Scalar &start,
    const at::Scalar &end,
    const at::Scalar &step,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory)
{
    const at::TensorOptions opts = at::TensorOptions()
        .dtype(dtype)
        .layout(layout)
        .device(device)
        .pinned_memory(pin_memory);
    const at::Tensor cpu_out = at::arange(start, end, step, opts.device(at::kCPU));
    at::Tensor out = at::empty(
        cpu_out.sizes(),
        opts.device(c10::Device(c10::DeviceType::PrivateUse1, 0)));
    copy_cpu_tensor_to_nntile(cpu_out, out);
    return out;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("arange.start_step", TORCH_FN(torch_nntile::arange_start_step));
    m.impl(
        "arange.start_out",
        TORCH_FN(torch_nntile::arange_start_step_out));
}
