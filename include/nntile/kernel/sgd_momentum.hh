/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sgd_momentum.hh
 * SGD_MOMENTUM low-level kernels
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/sgd_momentum/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/sgd_momentum/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::sgd_momentum
/*! Low-level implementations of SGD_MOMENTUM operation
 * */
namespace nntile::kernel::sgd_momentum
{

} // namespace nntile::kernel::sgd_momentum
