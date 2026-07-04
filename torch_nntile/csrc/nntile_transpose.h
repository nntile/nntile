/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_transpose.h
 */

#pragma once

#include <ATen/ATen.h>

namespace torch_nntile
{

at::Tensor model_transpose_forward(
    const at::Tensor &x,
    int64_t model_ndim);

at::Tensor model_transpose_backward(
    const at::Tensor &grad_out,
    int64_t model_ndim);

} // namespace torch_nntile
