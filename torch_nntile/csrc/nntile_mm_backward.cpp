/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_mm_backward.cpp
 *
 * PyTorch autograd for aten::mm decomposes into mm/bmm calls in the backward
 * graph; this module provides shared backward GEMM helpers for tests and future
 * explicit backward kernels.
 */

#include "nntile_executor.h"
#include "nntile_gemm_layout.h"
#include "nntile_graph_recorder_impl.h"

#include <ATen/Functions.h>
#include <torch/library.h>

namespace torch_nntile
{

void run_mm_backward_grad_a(
    const at::Tensor &grad_out,
    const at::Tensor &b,
    at::Tensor &grad_a,
    const GemmParams &forward_params)
{
    const PreparedGemmOperands prepared = prepare_mm_operands(grad_out, b);
    GemmParams params = infer_mm_backward_grad_a_params(forward_params);
    pin_graph_op_inputs({prepared.a, prepared.b});
    pin_graph_op_output(grad_a, false);
    tensor_gemm_fp32(
        params,
        prepared.a.data_ptr<float>(),
        prepared.a_gemm_shape,
        prepared.b.data_ptr<float>(),
        prepared.b_gemm_shape,
        grad_a.data_ptr<float>(),
        pytorch_sizes_vector(grad_a.sizes()));
}

void run_mm_backward_grad_b(
    const at::Tensor &a,
    const at::Tensor &grad_out,
    at::Tensor &grad_b,
    const GemmParams &forward_params)
{
    const PreparedGemmOperands prepared = prepare_mm_operands(a, grad_out);
    GemmParams params = infer_mm_backward_grad_b_params(forward_params);
    pin_graph_op_inputs({prepared.a, prepared.b});
    pin_graph_op_output(grad_b, false);
    tensor_gemm_fp32(
        params,
        prepared.a.data_ptr<float>(),
        prepared.a_gemm_shape,
        prepared.b.data_ptr<float>(),
        prepared.b_gemm_shape,
        grad_b.data_ptr<float>(),
        pytorch_sizes_vector(grad_b.sizes()));
}

} // namespace torch_nntile
