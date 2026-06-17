/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_executor.h
 */

#pragma once

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

} // namespace torch_nntile
