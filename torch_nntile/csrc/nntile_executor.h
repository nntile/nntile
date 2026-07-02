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

void tensor_linear_fp32(
    const at::Tensor &input,
    const at::Tensor &weight,
    at::Tensor &out);

void tensor_relu_fp32(const at::Tensor &input, at::Tensor &out);

void tensor_relu_backward_fp32(
    const at::Tensor &x,
    const at::Tensor &dy,
    at::Tensor &dx);

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

} // namespace torch_nntile
