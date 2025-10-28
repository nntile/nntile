/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/lamb_step.hh
 * Fused LAMB step
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/lamb_step/cpu.hh>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/lamb_step/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::lamb_step
/*! Low-level implementations of fused LAMB step
 * */
namespace nntile::kernel::lamb_step
{

} // namespace nntile::kernel::lamb_step
