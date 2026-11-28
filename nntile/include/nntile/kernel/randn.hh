/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/kernel/randn.hh
 * Randn low-level kernels
 *
 * @version 2.0.0
 * */

#pragma once

#include <nntile/kernel/randn/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/randn/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::randn
/*! Low-level implementations of Randn operation
 * */
namespace nntile::kernel::randn
{

} // namespace nntile::kernel::randn
