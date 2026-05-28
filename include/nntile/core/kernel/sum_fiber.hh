/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sum_fiber.hh
 * Sums over slices into a fiber of a buffer
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/core/kernel/sum_fiber/cpu.hh>
#include <nntile/core/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/core/kernel/sum_fiber/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::core::kernel::sum_fiber
/*! Low-level implementations of computing sum_fiber operation
 * */
namespace nntile::core::kernel::sum_fiber
{

} // namespace nntile::core::kernel::sum_fiber
