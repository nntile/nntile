/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_rope.h
 */

#pragma once

#include <ATen/ATen.h>

#include <array>

namespace torch_nntile
{

at::Tensor rope_forward(
    const at::Tensor &sin,
    const at::Tensor &cos,
    const at::Tensor &x);

at::Tensor rope_backward(
    const at::Tensor &sin,
    const at::Tensor &cos,
    const at::Tensor &grad_out,
    std::array<bool, 1> output_mask);

} // namespace torch_nntile
