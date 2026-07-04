/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_gemm_layout.h
 * Map PyTorch tensor layouts to NNTile GEMM parameters.
 */

#pragma once

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <tuple>
#include <vector>

namespace torch_nntile
{

struct GemmMatrixLayout
{
    std::vector<int64_t> gemm_shape;
    bool trans = false;
    bool needs_copy = false;
};

struct GemmParams
{
    bool trans_a = false;
    bool trans_b = false;
    int64_t ndim = 1;
    int64_t batch_ndim = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
};

struct PreparedGemmOperands
{
    at::Tensor a;
    at::Tensor b;
    GemmParams params;
    std::vector<int64_t> a_gemm_shape;
    std::vector<int64_t> b_gemm_shape;
    std::vector<int64_t> out_shape;
};

GemmMatrixLayout analyze_matrix_layout_for_nntile(const at::Tensor &tensor);

GemmMatrixLayout analyze_batched_gemm_operand_layout(const at::Tensor &tensor);

GemmMatrixLayout layout_from_nd_contiguous(const at::Tensor &tensor);

std::vector<int64_t> gemm_output_shape_pytorch(
    const std::vector<int64_t> &a_shape,
    const std::vector<int64_t> &b_shape,
    const GemmParams &params);

PreparedGemmOperands prepare_mm_operands(const at::Tensor &a, const at::Tensor &b);

PreparedGemmOperands prepare_bmm_operands(const at::Tensor &a, const at::Tensor &b);

PreparedGemmOperands prepare_linear_operands(
    const at::Tensor &input,
    const at::Tensor &weight);

//! Infer ``ndim`` / ``batch_ndim`` from NNTile GEMM shape rules (``batch_ndim=0``
//! free/broadcast axes on A, shared leading batch when sizes match).
std::pair<int64_t, int64_t> infer_gemm_params(
    c10::IntArrayRef a_shape,
    c10::IntArrayRef b_shape);

PreparedGemmOperands prepare_gemm_operands(
    const at::Tensor &a,
    const at::Tensor &b,
    int64_t ndim,
    int64_t batch_ndim);

PreparedGemmOperands prepare_gemm_operands_inferred(
    const at::Tensor &a,
    const at::Tensor &b);

GemmParams infer_gemm_backward_grad_a_params(
    const GemmParams &forward,
    int64_t b_rank);

GemmParams infer_gemm_backward_grad_b_params(
    const GemmParams &forward,
    int64_t a_rank);

GemmParams infer_mm_backward_grad_a_params(const GemmParams &forward);
GemmParams infer_mm_backward_grad_b_params(const GemmParams &forward);

GemmParams infer_linear_backward_grad_input_params(const GemmParams &forward);
GemmParams infer_linear_backward_grad_weight_params(const GemmParams &forward);

std::vector<int64_t> pytorch_sizes_vector(c10::IntArrayRef sizes);

void run_mm_backward_grad_a(
    const at::Tensor &grad_out,
    const at::Tensor &b,
    at::Tensor &grad_a,
    const GemmParams &forward_params);

void run_mm_backward_grad_b(
    const at::Tensor &a,
    const at::Tensor &grad_out,
    at::Tensor &grad_b,
    const GemmParams &forward_params);

} // namespace torch_nntile
