/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_executor.h
 */

#pragma once

#include <ATen/Tensor.h>
#include <c10/util/ArrayRef.h>

namespace torch_nntile
{

void tensor_add_fp32(
    float alpha,
    const at::Tensor &x,
    float beta,
    const at::Tensor &y,
    at::Tensor &out);

void tensor_add_inplace_fp32(
    float alpha,
    const at::Tensor &other,
    float beta,
    at::Tensor &self);

void tensor_mul_fp32(
    const at::Tensor &self,
    const at::Tensor &other,
    at::Tensor &out);

void tensor_mul_inplace_fp32(const at::Tensor &other, at::Tensor &self);

void tensor_linear_fp32(
    const at::Tensor &input,
    const at::Tensor &weight,
    at::Tensor &out);

void tensor_relu_fp32(const at::Tensor &input, at::Tensor &out);

void tensor_relu_backward_fp32(
    const at::Tensor &x,
    const at::Tensor &dy,
    at::Tensor &dx);

void tensor_silu_fp32(const at::Tensor &input, at::Tensor &out);

void tensor_silu_inplace_fp32(at::Tensor &self);

void tensor_silu_backward_fp32(
    const at::Tensor &x,
    const at::Tensor &dy,
    at::Tensor &dx);

void tensor_gelu_fp32(
    const at::Tensor &input,
    at::Tensor &out,
    bool approximate_tanh);

void tensor_gelu_inplace_fp32(at::Tensor &self, bool approximate_tanh);

void tensor_gelu_backward_fp32(
    const at::Tensor &x,
    const at::Tensor &dy,
    at::Tensor &dx,
    bool approximate_tanh);

void tensor_mm_fp32(
    const at::Tensor &a,
    const at::Tensor &b,
    at::Tensor &out);

void tensor_linear_backward_input_fp32(
    const at::Tensor &grad_out,
    const at::Tensor &weight,
    at::Tensor &grad_input);

void tensor_linear_backward_weight_fp32(
    const at::Tensor &grad_out,
    const at::Tensor &input,
    at::Tensor &grad_weight);

void tensor_cross_entropy_forward_fp32(
    const at::Tensor &logits,
    const at::Tensor &labels,
    std::int64_t ignore_index,
    bool mean_reduction,
    at::Tensor &loss);

void tensor_cross_entropy_backward_fp32(
    const at::Tensor &logits,
    const at::Tensor &labels,
    const at::Tensor &grad_output,
    at::Tensor &grad_row,
    at::Tensor &grad_logits,
    std::int64_t ignore_index,
    bool mean_reduction);

void tensor_sgd_step_fp32(
    int64_t num_iter,
    float momentum,
    float lr,
    float weight_decay,
    float dampening,
    bool nesterov,
    const at::Tensor &grad,
    at::Tensor &velocity,
    at::Tensor &param);

void tensor_adam_step_fp32(
    int64_t num_iter,
    float beta_1,
    float beta_2,
    float eps,
    float lr,
    float weight_decay,
    const at::Tensor &grad,
    at::Tensor &first_moment,
    at::Tensor &second_moment,
    at::Tensor &param);

void tensor_adamw_step_fp32(
    int64_t num_iter,
    float beta_1,
    float beta_2,
    float eps,
    float lr,
    float weight_decay,
    const at::Tensor &grad,
    at::Tensor &first_moment,
    at::Tensor &second_moment,
    at::Tensor &param);

void tensor_layer_norm_forward_fp32(
    const at::Tensor &input,
    const at::Tensor *weight,
    const at::Tensor *bias,
    bool has_weight,
    bool has_bias,
    at::Tensor &output,
    at::Tensor &mean,
    at::Tensor &rstd,
    int64_t norm_axis,
    float eps);

void tensor_layer_norm_backward_fp32(
    const at::Tensor &grad_out,
    const at::Tensor &input,
    const at::Tensor &mean,
    const at::Tensor &rstd,
    const at::Tensor *weight,
    bool has_weight,
    bool has_bias,
    at::Tensor *grad_input,
    at::Tensor *grad_weight,
    at::Tensor *grad_bias,
    bool grad_input_needed,
    bool grad_weight_needed,
    bool grad_bias_needed,
    int64_t norm_axis);

void tensor_rms_norm_forward_fp32(
    const at::Tensor &input,
    const at::Tensor *weight,
    bool has_weight,
    at::Tensor &output,
    at::Tensor &rstd,
    int64_t norm_axis,
    float eps);

void tensor_rms_norm_backward_fp32(
    const at::Tensor &grad_out,
    const at::Tensor &input,
    const at::Tensor &rstd,
    const at::Tensor *weight,
    bool has_weight,
    at::Tensor *grad_input,
    at::Tensor *grad_weight,
    bool grad_input_needed,
    bool grad_weight_needed,
    int64_t norm_axis);

} // namespace torch_nntile
