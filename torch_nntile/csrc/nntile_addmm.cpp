/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_addmm.cpp
 */

#include "nntile_broadcast.h"
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

at::Tensor broadcast_addmm_self(
    const at::Tensor &self,
    c10::IntArrayRef target_size)
{
    if (self.sizes().equals(target_size))
    {
        TORCH_CHECK(
            self.is_contiguous(),
            "nntile addmm: self must be contiguous");
        return self;
    }
    TORCH_CHECK(
        self.is_contiguous(),
        "nntile addmm: self must be contiguous before broadcast");
    const int64_t target_ndim = static_cast<int64_t>(target_size.size());
    const int64_t tensor_ndim = static_cast<int64_t>(self.sizes().size());
    TORCH_CHECK(
        tensor_ndim <= target_ndim,
        "nntile addmm: self rank exceeds target rank");
    std::vector<int64_t> repeats(
        static_cast<std::size_t>(target_ndim),
        1);
    const int64_t pad = target_ndim - tensor_ndim;
    for (int64_t d = 0; d < pad; ++d)
    {
        const int64_t out_dim = target_size[static_cast<std::size_t>(d)];
        TORCH_CHECK(
            out_dim >= 1,
            "nntile addmm: invalid target dimension");
        repeats[static_cast<std::size_t>(d)] = out_dim;
    }
    for (int64_t i = 0; i < tensor_ndim; ++i)
    {
        const int64_t in_dim = self.sizes()[static_cast<std::size_t>(i)];
        const int64_t out_dim = target_size[static_cast<std::size_t>(i + pad)];
        TORCH_CHECK(
            in_dim == 1 || in_dim == out_dim,
            "nntile addmm: self is not broadcastable to output shape");
        TORCH_CHECK(
            out_dim % in_dim == 0,
            "nntile addmm: output size is not divisible by self size");
        repeats[static_cast<std::size_t>(i + pad)] = out_dim / in_dim;
    }
    at::Tensor out = at::empty(
        target_size,
        self.options().memory_format(at::MemoryFormat::Contiguous));
    pin_graph_op_inputs({self});
    pin_graph_op_output(out, true);
    tensor_repeat_fp32(self, out, repeats);
    return out;
}

void run_addmm(
    const at::Tensor &self,
    const PreparedGemmOperands &prepared,
    const at::Scalar &beta,
    const at::Scalar &alpha,
    at::Tensor &out)
{
    pin_graph_op_inputs({prepared.a, prepared.b});
    pin_graph_op_output(out, false);

    GemmParams params = prepared.params;
    params.alpha = alpha.to<float>();
    params.beta = beta.to<float>();

    at::Tensor self_expanded = self;
    if (self.sizes() != prepared.out_shape)
    {
        self_expanded = broadcast_addmm_self(self, prepared.out_shape);
    }
    else
    {
        TORCH_CHECK(
            self_expanded.is_contiguous(),
            "nntile addmm: self must be contiguous");
    }

    if (params.beta == 0.0f)
    {
        tensor_gemm_fp32(
            params,
            prepared.a,
            prepared.a_gemm_shape,
            prepared.b,
            prepared.b_gemm_shape,
            out,
            prepared.out_shape);
        return;
    }

    pin_graph_op_inputs({self_expanded});
    tensor_gemm_accumulate_fp32(
        params,
        prepared.a,
        prepared.a_gemm_shape,
        prepared.b,
        prepared.b_gemm_shape,
        self_expanded,
        prepared.out_shape,
        out,
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
