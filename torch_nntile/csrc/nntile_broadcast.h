/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_broadcast.h
 * Broadcast helpers built from chained ``scale_slice`` ops.
 */

#pragma once

#include <c10/util/ArrayRef.h>

namespace torch_nntile
{

void tensor_repeat_fp32(
    const float *input_data,
    float *out_data,
    c10::IntArrayRef input_shape,
    c10::IntArrayRef repeats,
    c10::IntArrayRef out_shape);

} // namespace torch_nntile
