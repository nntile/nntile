/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_adam_step.h
 */

#pragma once

#include <ATen/Tensor.h>
#include <cstdint>

namespace torch_nntile
{

void adam_step(
    at::Tensor &param,
    const at::Tensor &grad,
    at::Tensor &first_moment,
    at::Tensor &second_moment,
    int64_t num_iter,
    double lr,
    double beta_1,
    double beta_2,
    double eps,
    double weight_decay);

void adamw_step(
    at::Tensor &param,
    const at::Tensor &grad,
    at::Tensor &first_moment,
    at::Tensor &second_moment,
    int64_t num_iter,
    double lr,
    double beta_1,
    double beta_2,
    double eps,
    double weight_decay);

} // namespace torch_nntile
