/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_rms_norm.h
 */

#pragma once

#include <array>
#include <ATen/Tensor.h>
#include <cstdint>
#include <optional>

namespace torch_nntile
{

std::tuple<at::Tensor, at::Tensor> rms_norm_forward(
    const at::Tensor &input,
    at::IntArrayRef normalized_shape,
    const std::optional<at::Tensor> &weight,
    std::optional<double> eps);

std::tuple<at::Tensor, at::Tensor> rms_norm_backward(
    const at::Tensor &grad_out,
    const at::Tensor &input,
    at::IntArrayRef normalized_shape,
    const at::Tensor &rstd,
    const std::optional<at::Tensor> &weight,
    std::array<bool, 2> output_mask);

} // namespace torch_nntile
