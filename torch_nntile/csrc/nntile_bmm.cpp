/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_bmm.cpp
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

void check_bmm_tensors(
    const at::Tensor &self,
    const at::Tensor &mat2,
    const std::optional<at::Tensor> &out = std::nullopt)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) && is_nntile_device(mat2.device()),
        "nntile bmm expects nntile tensors");
    if (out.has_value())
    {
        TORCH_CHECK(
            is_nntile_device(out->device()),
            "nntile bmm.out expects nntile output");
    }
    TORCH_CHECK(self.dim() == 3 && mat2.dim() == 3, "nntile bmm supports 3D only");
    TORCH_CHECK(
        self.size(0) == mat2.size(0),
        "nntile bmm: batch dimension mismatch");
    TORCH_CHECK(
        self.size(2) == mat2.size(1),
        "nntile bmm: inner dimension mismatch");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float &&
            mat2.scalar_type() == at::ScalarType::Float,
        "nntile bmm supports float32 only");
}

at::Tensor make_bmm_output(const at::Tensor &self, const at::Tensor &mat2)
{
    return at::empty(
        {self.size(0), self.size(1), mat2.size(2)},
        self.options().memory_format(at::MemoryFormat::Contiguous));
}

void run_bmm(const at::Tensor &self, const at::Tensor &mat2, at::Tensor &out)
{
    tensor_bmm_fp32(self, mat2, out);
}

} // namespace

at::Tensor bmm(const at::Tensor &self, const at::Tensor &mat2)
{
    nntile::GraphFillScope record;
    check_bmm_tensors(self, mat2);
    at::Tensor out = make_bmm_output(self, mat2);
    run_bmm(self, mat2, out);
    return out;
}

at::Tensor &bmm_out(const at::Tensor &self, const at::Tensor &mat2, at::Tensor &out)
{
    nntile::GraphFillScope record;
    check_bmm_tensors(self, mat2, out);
    TORCH_CHECK(
        out.sizes() ==
            at::IntArrayRef({self.size(0), self.size(1), mat2.size(2)}),
        "nntile bmm.out: output shape mismatch");
    run_bmm(self, mat2, out);
    return out;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("bmm", TORCH_FN(torch_nntile::bmm));
    m.impl("bmm.out", TORCH_FN(torch_nntile::bmm_out));
}
