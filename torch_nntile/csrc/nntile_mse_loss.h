/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_mse_loss.h
 */

#pragma once

#include <ATen/ATen.h>

namespace torch_nntile
{

//! ``loss = scale * ||x||^2`` (scalar).
at::Tensor mse_loss_forward(const at::Tensor &x, double scale);

//! ``grad_x = 2 * scale * x`` when ``needs_grad``; else undefined.
at::Tensor mse_loss_backward(
    const at::Tensor &x,
    double scale,
    bool needs_grad);

} // namespace torch_nntile
