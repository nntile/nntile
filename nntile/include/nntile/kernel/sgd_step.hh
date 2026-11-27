/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sgd_step.hh
 * Fused SGD with momentum step
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/sgd_step/cpu.hh>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/sgd_step/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::sgd_step
/*! Low-level implementations of fused SGD with momentum step
 * */
namespace nntile::kernel::sgd_step
{

} // namespace nntile::kernel::sgd_step
