/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_context.h
 */

#pragma once

#include <torch_nntile/runtime.hh>

namespace torch_nntile
{

void ensure_nntile_context();

//! True when libnntile was built on the NNHaul runtime backend
//! (NNTILE_USE_NNHAUL) instead of StarPU.
bool uses_nnhaul();

} // namespace torch_nntile
