/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_cross_entropy.h
 */

#pragma once

#include <ATen/Tensor.h>
#include <cstdint>

namespace torch_nntile
{

at::Tensor cross_entropy_forward(
    const at::Tensor &logits,
    const at::Tensor &target,
    int64_t reduction,
    int64_t ignore_index);

at::Tensor cross_entropy_backward(
    const at::Tensor &logits,
    const at::Tensor &target,
    const at::Tensor &grad_output,
    int64_t reduction,
    int64_t ignore_index);

} // namespace torch_nntile
