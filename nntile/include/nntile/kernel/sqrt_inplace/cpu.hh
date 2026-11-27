/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/kernel/sqrt_inplace/cpu.hh
 * Inplace sqrt operation on CPU
 *
 * @version 1.1.0
 * */
/*!
 * INTENTIONALLY DISABLED in the NNTile 2.0 transformer-only scope.
 * Kept in-tree for future re-enablement. No product path calls this code;
 * its tests are not registered while the family is in
 * NNTILE_DISABLED_KERNELS.
 */
#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::sqrt_inplace
{

// Inplace sqrt operation on a buffer
template<typename T>
void cpu(Index nelems, T *data)
    noexcept;

} // namespace nntile::kernel::sqrt_inplace
