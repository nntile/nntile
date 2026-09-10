/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_addmm.cpp
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

void check_addmm_tensors(
    const at::Tensor &self,
    const at::Tensor &mat1,
    const at::Tensor &mat2,
    const std::optional<at::Tensor> &out = std::nullopt)
{
    TORCH_CHECK(
        is_nntile_device(self.device()) &&
            is_nntile_device(mat1.device()) &&
            is_nntile_device(mat2.device()),
        "nntile addmm expects nntile tensors");
    if (out.has_value())
    {
        TORCH_CHECK(
            is_nntile_device(out->device()),
            "nntile addmm.out expects nntile output");
    }
    TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "nntile addmm supports 2D only");
    TORCH_CHECK(
        mat1.size(1) == mat2.size(0),
        "nntile addmm: inner dimension mismatch");
    TORCH_CHECK(
        mat1.scalar_type() == at::ScalarType::Float &&
            mat2.scalar_type() == at::ScalarType::Float &&
            self.scalar_type() == at::ScalarType::Float,
        "nntile addmm supports float32 only");
}

at::Tensor make_addmm_output(const at::Tensor &mat1, const at::Tensor &mat2)
{
    return at::empty(
        {mat1.size(0), mat2.size(1)},
        mat1.options().memory_format(at::MemoryFormat::Contiguous));
}

void run_addmm(
    const at::Tensor &self,
    const at::Tensor &mat1,
    const at::Tensor &mat2,
    const at::Scalar &beta,
    const at::Scalar &alpha,
    at::Tensor &out)
{
    tensor_addmm_fp32(
        self,
        mat1,
        mat2,
        beta.to<float>(),
        alpha.to<float>(),
        out);
}

} // namespace

at::Tensor addmm(
    const at::Tensor &self,
    const at::Tensor &mat1,
    const at::Tensor &mat2,
    const at::Scalar &beta,
    const at::Scalar &alpha)
{
    nntile::GraphFillScope record;
    check_addmm_tensors(self, mat1, mat2);
    at::Tensor out = make_addmm_output(mat1, mat2);
    run_addmm(self, mat1, mat2, beta, alpha, out);
    return out;
}

at::Tensor &addmm_out(
    const at::Tensor &self,
    const at::Tensor &mat1,
    const at::Tensor &mat2,
    const at::Scalar &beta,
    const at::Scalar &alpha,
    at::Tensor &out)
{
    nntile::GraphFillScope record;
    check_addmm_tensors(self, mat1, mat2, out);
    TORCH_CHECK(
        out.sizes() == at::IntArrayRef({mat1.size(0), mat2.size(1)}),
        "nntile addmm.out: output shape mismatch");
    run_addmm(self, mat1, mat2, beta, alpha, out);
    return out;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("addmm", TORCH_FN(torch_nntile::addmm));
    m.impl("addmm.out", TORCH_FN(torch_nntile::addmm_out));
}
