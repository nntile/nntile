/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/isfinite.hh
 * Accumulate flags for Inf and NaN values low-level kernel
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/core/kernel/isfinite/cpu.hh>
#include <nntile/core/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/core/kernel/isfinite/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::core::kernel::isfinite
/*! Low-level implementations of accumulate Inf and NaN flags
 * */
namespace nntile::core::kernel::isfinite
{

} // namespace nntile::core::kernel::isfinite
