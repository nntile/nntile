/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_mm.cpp
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

void check_mm_tensors(
    const at::Tensor &self,
    const at::Tensor &mat2,
    const std::optional<at::Tensor> &out = std::nullopt)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) && is_nntile_device(mat2.device()),
        "nntile mm expects nntile tensors");
    if (out.has_value())
    {
        TORCH_CHECK(
            is_nntile_device(out->device()),
            "nntile mm.out expects nntile output");
    }
    TORCH_CHECK(self.dim() == 2 && mat2.dim() == 2, "nntile mm supports 2D only");
    TORCH_CHECK(
        self.size(1) == mat2.size(0),
        "nntile mm: inner dimension mismatch");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float &&
            mat2.scalar_type() == at::ScalarType::Float,
        "nntile mm supports float32 only");
    TORCH_CHECK(
        self.is_contiguous() && mat2.is_contiguous(),
        "nntile mm requires contiguous tensors");
}

at::Tensor make_mm_output(const at::Tensor &self, const at::Tensor &mat2)
{
    return at::empty(
        {self.size(0), mat2.size(1)},
        self.options().memory_format(at::MemoryFormat::Contiguous));
}

void run_mm(const at::Tensor &self, const at::Tensor &mat2, at::Tensor &out)
{
    pin_graph_op_inputs({self, mat2});
    pin_graph_op_output(out, false);
    tensor_mm_fp32(self, mat2, out);
}

} // namespace

at::Tensor mm(const at::Tensor &self, const at::Tensor &mat2)
{
    check_mm_tensors(self, mat2);
    at::Tensor out = make_mm_output(self, mat2);
    run_mm(self, mat2, out);
    return out;
}

at::Tensor &mm_out(const at::Tensor &self, const at::Tensor &mat2, at::Tensor &out)
{
    check_mm_tensors(self, mat2, out);
    TORCH_CHECK(
        out.sizes() == make_mm_output(self, mat2).sizes(),
        "nntile mm.out: output shape mismatch");
    TORCH_CHECK(out.is_contiguous(), "nntile mm.out requires contiguous out");
    run_mm(self, mat2, out);
    return out;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("mm", TORCH_FN(torch_nntile::mm));
    m.impl("mm.out", TORCH_FN(torch_nntile::mm_out));
}
