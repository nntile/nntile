/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_broadcast.h
 * Broadcast helpers built from chained ``scale_slice`` ops.
 */

#pragma once

#include <ATen/Tensor.h>
#include <c10/util/ArrayRef.h>

namespace torch_nntile
{

void tensor_repeat_fp32(
    const at::Tensor &input,
    at::Tensor &out,
    c10::IntArrayRef repeats);

} // namespace torch_nntile
