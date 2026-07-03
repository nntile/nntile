/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_executor.h
 */

#pragma once

#include "nntile_gemm_layout.h"

#include <c10/util/ArrayRef.h>

namespace torch_nntile
{

void tensor_add_fp32(
    float alpha,
    const float *x_data,
    float beta,
    const float *y_data,
    float *out_data,
    c10::IntArrayRef pytorch_shape);

void tensor_add_inplace_fp32(
    float alpha,
    const float *other_data,
    float beta,
    float *self_data,
    c10::IntArrayRef pytorch_shape);

void tensor_mul_fp32(
    const float *self_data,
    const float *other_data,
    float *out_data,
    c10::IntArrayRef pytorch_shape);

void tensor_mul_inplace_fp32(
    const float *other_data,
    float *self_data,
    c10::IntArrayRef pytorch_shape);

void tensor_linear_fp32(
    const float *input_data,
    c10::IntArrayRef input_shape,
    const float *weight_data,
    c10::IntArrayRef weight_shape,
    float *out_data,
    c10::IntArrayRef out_shape);

void tensor_relu_fp32(
    const float *input_data,
    float *out_data,
    c10::IntArrayRef pytorch_shape);

void tensor_relu_backward_fp32(
    const float *x_data,
    const float *dy_data,
    float *dx_data,
    c10::IntArrayRef pytorch_shape);

void tensor_silu_fp32(
    const float *input_data,
    float *out_data,
    c10::IntArrayRef pytorch_shape);

void tensor_silu_inplace_fp32(
    float *data,
    c10::IntArrayRef pytorch_shape);

void tensor_silu_backward_fp32(
    const float *x_data,
    const float *dy_data,
    float *dx_data,
    c10::IntArrayRef pytorch_shape);

void tensor_gelu_fp32(
    const float *input_data,
    float *out_data,
    c10::IntArrayRef pytorch_shape,
    bool approximate_tanh);

void tensor_gelu_inplace_fp32(
    float *data,
    c10::IntArrayRef pytorch_shape,
    bool approximate_tanh);

void tensor_gelu_backward_fp32(
    const float *x_data,
    const float *dy_data,
    float *dx_data,
    c10::IntArrayRef pytorch_shape,
    bool approximate_tanh);

void tensor_gemm_fp32(
    const GemmParams &params,
    const float *a_data,
    c10::IntArrayRef a_gemm_shape,
    const float *b_data,
    c10::IntArrayRef b_gemm_shape,
    float *out_data,
    c10::IntArrayRef out_shape);

void tensor_gemm_accumulate_fp32(
    const GemmParams &params,
    const float *a_data,
    c10::IntArrayRef a_gemm_shape,
    const float *b_data,
    c10::IntArrayRef b_gemm_shape,
    const float *c_data,
    c10::IntArrayRef c_shape,
    float *out_data,
    c10::IntArrayRef out_shape);

void tensor_mm_fp32(
    const float *a_data,
    c10::IntArrayRef a_shape,
    const float *b_data,
    c10::IntArrayRef b_shape,
    float *out_data,
    c10::IntArrayRef out_shape);

void tensor_linear_backward_input_fp32(
    const float *grad_out_data,
    c10::IntArrayRef grad_out_shape,
    const float *weight_data,
    c10::IntArrayRef weight_shape,
    float *grad_input_data,
    c10::IntArrayRef grad_input_shape);

void tensor_linear_backward_weight_fp32(
    const float *grad_out_data,
    c10::IntArrayRef grad_out_shape,
    const float *input_data,
    c10::IntArrayRef input_shape,
    float *grad_weight_data,
    c10::IntArrayRef grad_weight_shape);

void tensor_cross_entropy_forward_fp32(
    const float *logits_data,
    c10::IntArrayRef logits_shape,
    const std::int64_t *labels_data,
    c10::IntArrayRef labels_shape,
    std::int64_t ignore_index,
    bool mean_reduction,
    float *loss_data);

void tensor_cross_entropy_backward_fp32(
    const float *logits_data,
    c10::IntArrayRef logits_shape,
    const std::int64_t *labels_data,
    c10::IntArrayRef labels_shape,
    const float *grad_output_data,
    float *grad_row_data,
    float *grad_logits_data,
    std::int64_t ignore_index,
    bool mean_reduction);

void tensor_sgd_step_fp32(
    int64_t num_iter,
    float momentum,
    float lr,
    float weight_decay,
    float dampening,
    bool nesterov,
    const float *grad_data,
    float *velocity_data,
    float *param_data,
    c10::IntArrayRef pytorch_shape);

void tensor_adam_step_fp32(
    int64_t num_iter,
    float beta_1,
    float beta_2,
    float eps,
    float lr,
    float weight_decay,
    const float *grad_data,
    float *first_moment_data,
    float *second_moment_data,
    float *param_data,
    c10::IntArrayRef pytorch_shape);

void tensor_adamw_step_fp32(
    int64_t num_iter,
    float beta_1,
    float beta_2,
    float eps,
    float lr,
    float weight_decay,
    const float *grad_data,
    float *first_moment_data,
    float *second_moment_data,
    float *param_data,
    c10::IntArrayRef pytorch_shape);

} // namespace torch_nntile
