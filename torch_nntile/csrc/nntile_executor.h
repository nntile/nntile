/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_executor.h
 */

#pragma once

#include "nntile_gemm_layout.h"

#include <ATen/Tensor.h>
#include <c10/util/ArrayRef.h>

#include <cstdint>
#include <vector>

#ifdef TORCH_NNTILE_USE_LIBNNTILE
#include <nntile/base_types.hh>
#else
namespace nntile
{
using Index = std::int64_t;
} // namespace nntile
#endif

namespace torch_nntile
{

void tensor_add_fp32(
    float alpha,
    const at::Tensor &x,
    float beta,
    const at::Tensor &y,
    at::Tensor &out);

void tensor_model_transpose_forward_fp32(
    const at::Tensor &src,
    at::Tensor &dst,
    int64_t model_ndim);

void tensor_model_transpose_backward_fp32(
    const at::Tensor &grad_out,
    at::Tensor &grad_src,
    int64_t model_ndim);

void tensor_swap_two_axes_fp32(
    const at::Tensor &src,
    at::Tensor &dst,
    int64_t dim0,
    int64_t dim1);

void tensor_add_inplace_fp32(
    float alpha,
    const at::Tensor &other,
    float beta,
    at::Tensor &self);

void tensor_fill_fp32(at::Tensor &self, float value);

void tensor_mul_fp32(
    const at::Tensor &self,
    const at::Tensor &other,
    at::Tensor &out);

void tensor_mul_inplace_fp32(const at::Tensor &other, at::Tensor &self);

void tensor_hypot_fp32(
    const at::Tensor &self,
    const at::Tensor &other,
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

void tensor_gemm_fp32(
    const GemmParams &params,
    const at::Tensor &a,
    c10::IntArrayRef a_gemm_shape,
    const at::Tensor &b,
    c10::IntArrayRef b_gemm_shape,
    at::Tensor &out,
    c10::IntArrayRef out_shape);

void tensor_gemm_accumulate_fp32(
    const GemmParams &params,
    const at::Tensor &a,
    c10::IntArrayRef a_gemm_shape,
    const at::Tensor &b,
    c10::IntArrayRef b_gemm_shape,
    const at::Tensor &c,
    c10::IntArrayRef c_shape,
    at::Tensor &out,
    c10::IntArrayRef out_shape);

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

void tensor_linear_add_bias_fp32(
    at::Tensor &output,
    const at::Tensor &bias);

void tensor_linear_grad_bias_fp32(
    const at::Tensor &grad_output,
    at::Tensor &grad_bias);

//! ``out = alpha * fiber + beta * tensor`` (NNTile ``add_fiber``, no broadcast).
void tensor_add_fiber_fp32(
    float alpha,
    const at::Tensor &fiber,
    float beta,
    const at::Tensor &tensor,
    at::Tensor &out,
    int64_t axis,
    int64_t batch_ndim);

//! ``dst = alpha * sum_fiber(src)`` along ``axis`` (fiber grad for ``add_fiber``).
void tensor_sum_fiber_fp32(
    const at::Tensor &src,
    at::Tensor &dst,
    int64_t axis,
    int64_t batch_ndim,
    float alpha);

void tensor_cross_entropy_forward_fp32(
    const at::Tensor &logits,
    const at::Tensor &labels,
    std::int64_t ignore_index,
    bool mean_reduction,
    at::Tensor &loss,
    at::Tensor &maxsumexp);

void tensor_cross_entropy_backward_fp32(
    const at::Tensor &logits,
    const at::Tensor &labels,
    const at::Tensor &grad_output,
    const at::Tensor &maxsumexp,
    at::Tensor &grad_row,
    at::Tensor &grad_logits,
    std::int64_t ignore_index,
    bool mean_reduction);

void tensor_softmax_fp32(
    const at::Tensor &input,
    at::Tensor &out,
    int64_t dim);

void tensor_softmax_backward_fp32(
    const at::Tensor &grad_output,
    const at::Tensor &output,
    at::Tensor &grad_input,
    int64_t dim);

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

void tensor_norm_fp32(
    const at::Tensor &x,
    at::Tensor &out);

void tensor_norm_slice_fp32(
    const at::Tensor &x,
    at::Tensor &out,
    int64_t axis,
    bool keepdim);

void tensor_sum_dimlist_fp32(
    const at::Tensor &input,
    at::Tensor &out,
    at::OptionalIntArrayRef dim,
    bool keepdim);

void tensor_mul_scalar_fp32(
    const at::Tensor &input,
    at::Tensor &out,
    float scalar);

void tensor_cat_fp32(
    const std::vector<at::Tensor> &inputs,
    at::Tensor &out,
    int64_t dim);

void tensor_narrow_fp32(
    const at::Tensor &input,
    int64_t dim,
    int64_t start,
    int64_t length,
    at::Tensor &out);

void tensor_split_with_sizes_fp32(
    const at::Tensor &input,
    int64_t dim,
    const std::vector<int64_t> &split_sizes,
    const std::vector<at::Tensor> &outputs);

void tensor_embedding_forward_fp32(
    const at::Tensor &indices,
    const at::Tensor &weight,
    at::Tensor &out,
    nntile::Index axis);

void tensor_embedding_backward_fp32(
    const at::Tensor &indices,
    const at::Tensor &grad_out,
    at::Tensor &grad_weight,
    nntile::Index axis,
    int redux);

void tensor_sdpa_forward_fp32(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor *mask,
    at::Tensor &out,
    int64_t batch_ndim);

void tensor_sdpa_backward_fp32(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor *mask,
    const at::Tensor &grad_out,
    at::Tensor &grad_q,
    at::Tensor &grad_k,
    at::Tensor &grad_v,
    int64_t batch_ndim);

} // namespace torch_nntile
