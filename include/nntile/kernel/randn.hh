/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/randn.hh
 * Randn low-level kernels
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/randn/cpu.hh>
// No support for randn on CUDA
//#include <nntile/defs.h>
//#ifdef NNTILE_USE_CUDA
//#include <nntile/kernel/randn/cuda.hh>
//#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::randn
/*! Low-level implementations of Randn operation
 * */
namespace nntile::kernel::randn
{

} // namespace nntile::kernel::randn
