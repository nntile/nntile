/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_sgd_step.h
 */

#pragma once

#include <ATen/Tensor.h>
#include <cstdint>

namespace torch_nntile
{

void sgd_step(
    at::Tensor &param,
    at::Tensor &grad,
    at::Tensor &velocity,
    int64_t num_iter,
    double lr,
    double momentum,
    double weight_decay,
    double dampening,
    bool nesterov);

} // namespace torch_nntile
