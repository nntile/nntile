/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/scale_fiber.hh
 * Per-element scaling of a tensor with a broadcasted fiber
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/scale_fiber/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/scale_fiber/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::scale_fiber
/*! Low-level implementations of scale_fiber operation
 * */
namespace nntile::kernel::scale_fiber
{

} // namespace nntile::kernel::scale_fiber
