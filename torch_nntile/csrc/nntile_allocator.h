/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_allocator.h
 * Host std::vector-backed allocator for PrivateUse1.
 */

#pragma once

#include <c10/core/Allocator.h>

#include <cstdint>

namespace torch_nntile
{

c10::Allocator *get_nntile_allocator();

//! Number of host storage buffers released (for GC investigation).
std::int64_t storage_release_count();

void reset_storage_release_count();

} // namespace torch_nntile
