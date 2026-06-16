/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_cpu_fallback.h
 */

#pragma once

#include <torch/library.h>

namespace torch_nntile
{

void cpu_fallback(const c10::OperatorHandle &op, torch::jit::Stack *stack);

} // namespace torch_nntile
