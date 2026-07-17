/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_gemm.cpp
 */

#include "nntile_gemm.h"

#include "nntile_executor.h"
#include "nntile_gemm_layout.h"
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

void check_gemm_tensors(
    const at::Tensor &a,
    const at::Tensor &b)
{
    TORCH_CHECK(
        is_nntile_device(a.device()) && is_nntile_device(b.device()),
        "nntile gemm expects nntile tensors");
    TORCH_CHECK(a.dim() >= 1 && b.dim() >= 1, "nntile gemm: operands must be >= 1D");
    TORCH_CHECK(
        a.scalar_type() == at::ScalarType::Float &&
            b.scalar_type() == at::ScalarType::Float,
        "nntile gemm supports float32 only");
}

at::Tensor make_gemm_output(
    const std::vector<int64_t> &out_shape,
    const at::Tensor &ref)
{
    std::vector<int64_t> sizes(out_shape.begin(), out_shape.end());
    return at::empty(
        sizes,
        ref.options().memory_format(at::MemoryFormat::Contiguous));
}

void run_gemm(const PreparedGemmOperands &prepared, at::Tensor &out)
{
    tensor_gemm_fp32(
        prepared.params,
        prepared.a,
        prepared.a_gemm_shape,
        prepared.b,
        prepared.b_gemm_shape,
        out,
        prepared.out_shape);
}

} // namespace

at::Tensor gemm_forward(
    const at::Tensor &a,
    const at::Tensor &b,
    int64_t ndim,
    int64_t batch_ndim,
    bool trans_a,
    bool trans_b)
{
    check_gemm_tensors(a, b);
    const PreparedGemmOperands prepared =
        prepare_gemm_operands(a, b, ndim, batch_ndim, trans_a, trans_b);
    at::Tensor out = make_gemm_output(prepared.out_shape, a);
    run_gemm(prepared, out);
    return out;
}

std::tuple<at::Tensor, at::Tensor> gemm_backward(
    const at::Tensor &a,
    const at::Tensor &b,
    const at::Tensor &grad_out,
    int64_t ndim,
    int64_t batch_ndim,
    std::array<bool, 2> output_mask,
    bool trans_a,
    bool trans_b)
{
    check_gemm_tensors(a, b);
    TORCH_CHECK(
        is_nntile_device(grad_out.device()),
        "nntile gemm_backward expects nntile grad_out");
    TORCH_CHECK(
        grad_out.scalar_type() == at::ScalarType::Float,
        "nntile gemm_backward supports float32 only");

    const PreparedGemmOperands forward =
        prepare_gemm_operands(a, b, ndim, batch_ndim, trans_a, trans_b);
    const GemmMatrixLayout grad_out_layout = layout_from_nd_contiguous(grad_out);
    TORCH_CHECK(
        !grad_out_layout.needs_copy,
        "nntile gemm_backward: grad_out must be contiguous");
    const at::Tensor &grad_out_prepared = grad_out;

    // Match ``nntile::NNGemmOp::backward`` operand order / transpose flags.
    at::Tensor grad_a;
    at::Tensor grad_b;
    if (output_mask[0])
    {
        const int64_t b_rank =
            static_cast<int64_t>(forward.b_gemm_shape.size());
        GemmParams params;
        params.ndim = b_rank - forward.params.batch_ndim - forward.params.ndim;
        params.batch_ndim = forward.params.batch_ndim;
        grad_a = at::empty_like(a);
        if (!forward.params.trans_a)
        {
            params.trans_a = false;
            params.trans_b = !forward.params.trans_b;
            tensor_gemm_fp32(
                params,
                grad_out_prepared,
                grad_out_layout.gemm_shape,
                forward.b,
                forward.b_gemm_shape,
                grad_a,
                pytorch_sizes_vector(grad_a.sizes()));
        }
        else
        {
            params.trans_a = forward.params.trans_b;
            params.trans_b = true;
            tensor_gemm_fp32(
                params,
                forward.b,
                forward.b_gemm_shape,
                grad_out_prepared,
                grad_out_layout.gemm_shape,
                grad_a,
                pytorch_sizes_vector(grad_a.sizes()));
        }
    }
    if (output_mask[1])
    {
        const int64_t a_rank =
            static_cast<int64_t>(forward.a_gemm_shape.size());
        GemmParams params;
        params.ndim = a_rank - forward.params.batch_ndim - forward.params.ndim;
        params.batch_ndim = forward.params.batch_ndim;
        grad_b = at::empty_like(b);
        if (!forward.params.trans_b)
        {
            params.trans_a = !forward.params.trans_a;
            params.trans_b = false;
            tensor_gemm_fp32(
                params,
                forward.a,
                forward.a_gemm_shape,
                grad_out_prepared,
                grad_out_layout.gemm_shape,
                grad_b,
                pytorch_sizes_vector(grad_b.sizes()));
        }
        else
        {
            params.trans_a = true;
            params.trans_b = forward.params.trans_a;
            tensor_gemm_fp32(
                params,
                grad_out_prepared,
                grad_out_layout.gemm_shape,
                forward.a,
                forward.a_gemm_shape,
                grad_b,
                pytorch_sizes_vector(grad_b.sizes()));
        }
    }
    return {grad_a, grad_b};
}

at::Tensor matmul_nd(const at::Tensor &a, const at::Tensor &b)
{
    check_gemm_tensors(a, b);
    PreparedGemmOperands prepared;
    if (a.dim() == 2 && b.dim() == 2)
    {
        prepared = prepare_mm_operands(a, b);
    }
    else if (a.dim() == 3 && b.dim() == 3 && a.size(0) == b.size(0))
    {
        prepared = prepare_bmm_operands(a, b);
    }
    else
    {
        prepared = prepare_gemm_operands_inferred(a, b);
    }
    at::Tensor out = make_gemm_output(prepared.out_shape, a);
    run_gemm(prepared, out);
    return out;
}

std::tuple<at::Tensor, at::Tensor> matmul_backward(
    const at::Tensor &grad,
    const at::Tensor &self,
    const at::Tensor &other,
    std::array<bool, 2> mask)
{
    check_gemm_tensors(self, other);
    TORCH_CHECK(
        is_nntile_device(grad.device()),
        "nntile matmul_backward expects nntile grad");
    PreparedGemmOperands prepared;
    if (self.dim() == 2 && other.dim() == 2)
    {
        prepared = prepare_mm_operands(self, other);
    }
    else if (
        self.dim() == 3 && other.dim() == 3 &&
        self.size(0) == other.size(0))
    {
        prepared = prepare_bmm_operands(self, other);
    }
    else
    {
        prepared = prepare_gemm_operands_inferred(self, other);
    }
    return gemm_backward(
        prepared.a,
        prepared.b,
        grad.is_contiguous() ? grad : grad.contiguous(),
        prepared.params.ndim,
        prepared.params.batch_ndim,
        mask,
        prepared.params.trans_a,
        prepared.params.trans_b);
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("matmul", TORCH_FN(torch_nntile::matmul_nd));
    m.impl("matmul_backward", TORCH_FN(torch_nntile::matmul_backward));
}
