/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_add_fiber.h
 */

#pragma once

#include <ATen/ATen.h>

#include <array>
#include <tuple>

namespace torch_nntile
{

at::Tensor add_fiber_forward(
    const at::Tensor &fiber,
    const at::Tensor &tensor,
    int64_t axis,
    int64_t batch_ndim,
    double alpha = 1.0,
    double beta = 1.0);

std::tuple<at::Tensor, at::Tensor> add_fiber_backward(
    const at::Tensor &grad_out,
    const at::Tensor &fiber,
    const at::Tensor &tensor,
    int64_t axis,
    int64_t batch_ndim,
    std::array<bool, 2> output_mask,
    double alpha = 1.0,
    double beta = 1.0);

} // namespace torch_nntile
