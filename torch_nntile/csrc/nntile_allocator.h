/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_allocator.h
 * Host std::vector-backed allocator for PrivateUse1.
 */

#pragma once

#include <c10/core/Allocator.h>

namespace torch_nntile
{

c10::Allocator *get_nntile_allocator();

} // namespace torch_nntile
