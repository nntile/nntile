/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_addmm.cpp
 */

#include "nntile_executor.h"
#include "nntile_gemm_layout.h"
#include "nntile_graph_recorder_impl.h"

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
        mat1.scalar_type() == at::ScalarType::Float &&
            mat2.scalar_type() == at::ScalarType::Float &&
            self.scalar_type() == at::ScalarType::Float,
        "nntile addmm supports float32 only");
}

at::Tensor make_addmm_output(
    const std::vector<int64_t> &out_shape,
    const at::Tensor &ref)
{
    std::vector<int64_t> sizes(out_shape.begin(), out_shape.end());
    return at::empty(
        sizes,
        ref.options().memory_format(at::MemoryFormat::Contiguous));
}

void run_addmm(
    const at::Tensor &self,
    const PreparedGemmOperands &prepared,
    const at::Scalar &beta,
    const at::Scalar &alpha,
    at::Tensor &out)
{
    pin_graph_op_inputs({self, prepared.a, prepared.b});
    pin_graph_op_output(out, true);

    GemmParams params = prepared.params;
    params.alpha = alpha.to<float>();
    params.beta = beta.to<float>();

    at::Tensor self_expanded = self;
    if (self.sizes() != prepared.out_shape)
    {
        self_expanded = self.expand(prepared.out_shape);
    }
    if (!self_expanded.is_contiguous())
    {
        self_expanded = self_expanded.contiguous();
    }

    if (params.beta == 0.0f)
    {
        tensor_gemm_fp32(
            params,
            prepared.a.data_ptr<float>(),
            prepared.a_gemm_shape,
            prepared.b.data_ptr<float>(),
            prepared.b_gemm_shape,
            out.data_ptr<float>(),
            prepared.out_shape);
        return;
    }

    out.copy_(self_expanded);
    tensor_gemm_accumulate_fp32(
        params,
        prepared.a.data_ptr<float>(),
        prepared.a_gemm_shape,
        prepared.b.data_ptr<float>(),
        prepared.b_gemm_shape,
        out.data_ptr<float>(),
        prepared.out_shape,
        out.data_ptr<float>(),
        prepared.out_shape);
}

} // namespace

at::Tensor addmm(
    const at::Tensor &self,
    const at::Tensor &mat1,
    const at::Tensor &mat2,
    const at::Scalar &beta,
    const at::Scalar &alpha)
{
    check_addmm_tensors(self, mat1, mat2);
    const PreparedGemmOperands prepared = prepare_mm_operands(mat1, mat2);
    at::Tensor out = make_addmm_output(prepared.out_shape, mat1);
    run_addmm(self, prepared, beta, alpha, out);
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
    check_addmm_tensors(self, mat1, mat2, out);
    const PreparedGemmOperands prepared = prepare_mm_operands(mat1, mat2);
    TORCH_CHECK(
        out.sizes().vec() == prepared.out_shape,
        "nntile addmm.out: output shape mismatch");
    TORCH_CHECK(out.is_contiguous(), "nntile addmm.out requires contiguous out");
    run_addmm(self, prepared, beta, alpha, out);
    return out;
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("addmm", TORCH_FN(torch_nntile::addmm));
    m.impl("addmm.out", TORCH_FN(torch_nntile::addmm_out));
}
