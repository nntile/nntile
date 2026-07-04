/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/kernel/swap_two_axes/cpu.hh
 * Swap axes 1 and 3 in a 5D buffer on CPU.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::swap_two_axes
{

template<typename T>
void cpu(
    Index d0,
    Index d1,
    Index d2,
    Index d3,
    Index d4,
    const T *src,
    T *dst) noexcept;

} // namespace nntile::kernel::swap_two_axes
