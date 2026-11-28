/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/kernel_scope.hh
 * Build-time kernel scope: kernel families switched off for a
 * transformers-only product. The list comes from the configure-time
 * NNTILE_DISABLED_KERNELS cache variable; an empty list enables all.
 *
 * SPDX-License-Identifier: MIT
 *
 * @version 2.0.0
 * */

#pragma once

#include <string_view>

namespace nntile
{

//! True when this build disables the named kernel family.
bool kernel_family_disabled(std::string_view name) noexcept;

//! Throws std::runtime_error naming the family when it is disabled.
void require_kernel_family(std::string_view name);

} // namespace nntile
