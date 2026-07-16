/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_transpose.h
 * Umbrella: cyclic ``model_transpose`` + HF ``swap_two_axes``.
 *
 * Prefer including the specific header:
 * - ``nntile_model_transpose.h`` - native C++ models (cyclic)
 * - ``nntile_swap_two_axes.h`` - HF ATen ``transpose`` bridge only
 */

#pragma once

#include "nntile_model_transpose.h"
#include "nntile_swap_two_axes.h"
