/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sumprod_fiber.hh
 * Sums over slices into a fiber of a product of buffers
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/sumprod_fiber/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/sumprod_fiber/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::sumprod_fiber
/*! Low-level implementations of sumprod_fiber operation
 * axed
 * */
namespace nntile::kernel::sumprod_fiber
{

} // namespace nntile::kernel::sumprod_fiber
