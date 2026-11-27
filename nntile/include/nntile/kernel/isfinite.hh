/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/kernel/isfinite.hh
 * Accumulate flags for Inf and NaN values low-level kernel
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

#include <nntile/kernel/isfinite/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/isfinite/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::isfinite
/*! Low-level implementations of accumulate Inf and NaN flags
 * */
namespace nntile::kernel::isfinite
{

} // namespace nntile::kernel::isfinite
